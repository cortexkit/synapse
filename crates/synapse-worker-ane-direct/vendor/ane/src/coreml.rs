//! CoreML model loading and inference via MLModel.
//! Uses objc2 to call CoreML.framework natively — no Python needed at runtime.

use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject};
use objc2::{class, msg_send};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSString, NSURL};
use std::sync::Once;
use std::time::Instant;

static LOAD_COREML: Once = Once::new();

fn ensure_coreml_loaded() {
    LOAD_COREML.call_once(|| {
        unsafe {
            // Load CoreML.framework dynamically
            let path = c"/System/Library/Frameworks/CoreML.framework/CoreML";
            libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL);
        }
    });
}

/// A loaded CoreML model ready for inference.
pub struct CoreMLModel {
    model: Retained<AnyObject>,
}

unsafe impl Send for CoreMLModel {}

impl CoreMLModel {
    /// Load a CoreML model from a .mlpackage or .mlmodelc directory.
    pub fn load(path: &str) -> Result<Self, String> {
        ensure_coreml_loaded();
        let abs_path =
            std::fs::canonicalize(path).map_err(|e| format!("canonicalize {path}: {e}"))?;
        let path_str = abs_path.to_str().ok_or("invalid path")?;

        unsafe {
            // Create NSURL
            let ns_path = NSString::from_str(path_str);
            let url: Retained<NSURL> = msg_send![class!(NSURL), fileURLWithPath: &*ns_path];

            // Create MLModelConfiguration with ANE compute units
            let config_cls = AnyClass::get(c"MLModelConfiguration")
                .ok_or("MLModelConfiguration class not found — CoreML not available")?;
            let config: Retained<AnyObject> = msg_send![config_cls, new];
            // computeUnits = 2 (MLComputeUnitsAll)
            let _: () = msg_send![&*config, setComputeUnits: 2i64];

            // Load model
            let model_cls = AnyClass::get(c"MLModel").ok_or("MLModel class not found")?;
            let mut error: *mut NSError = std::ptr::null_mut();
            let model: *mut AnyObject = msg_send![
                model_cls,
                modelWithContentsOfURL: &*url
                configuration: &*config
                error: &mut error
            ];

            if model.is_null() {
                let desc = if !error.is_null() {
                    let desc: Retained<NSString> = msg_send![&*error, localizedDescription];
                    desc.to_string()
                } else {
                    "unknown error".to_string()
                };
                return Err(format!("MLModel load failed: {desc}"));
            }

            Ok(CoreMLModel {
                model: Retained::retain(model).unwrap(),
            })
        }
    }

    /// Run a single prediction with random input.
    /// Returns latency in microseconds.
    pub fn predict_once(&self, input_name: &str, shape: &[usize]) -> Result<i64, String> {
        let total_elements: usize = shape.iter().product();
        let data: Vec<f32> = vec![0.01; total_elements];

        unsafe {
            // Create MLMultiArray
            let shape_ns = shape_to_nsarray(shape);
            let multi_cls = AnyClass::get(c"MLMultiArray").ok_or("MLMultiArray class not found")?;
            let mut error: *mut NSError = std::ptr::null_mut();
            let multi_array: *mut AnyObject = msg_send![multi_cls, alloc];
            let multi_array: Retained<AnyObject> = {
                // dataType 65600 = Float32
                let obj: *mut AnyObject = msg_send![
                    multi_array,
                    initWithShape: &*shape_ns
                    dataType: 65600i64
                    error: &mut error
                ];
                if obj.is_null() {
                    return Err("MLMultiArray init failed".to_string());
                }
                Retained::retain(obj).unwrap()
            };

            // Copy data into MLMultiArray
            let ptr: *mut f32 = msg_send![&*multi_array, dataPointer];
            if !ptr.is_null() {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, total_elements);
            }

            // Create feature provider dictionary
            let key = NSString::from_str(input_name);
            let value_cls =
                AnyClass::get(c"MLFeatureValue").ok_or("MLFeatureValue class not found")?;
            let feature_value: Retained<AnyObject> = msg_send![
                value_cls,
                featureValueWithMultiArray: &*multi_array
            ];

            let provider_cls = AnyClass::get(c"MLDictionaryFeatureProvider")
                .ok_or("MLDictionaryFeatureProvider class not found")?;

            // Build NSDictionary with one key
            let keys: [&NSString; 1] = [&*key];
            let values: [&AnyObject; 1] = [&*feature_value];
            let keys_arr: Retained<NSArray<NSString>> = NSArray::from_slice(&keys);
            let vals_arr: Retained<NSArray<AnyObject>> = NSArray::from_slice(&values);
            let dict: Retained<NSDictionary<NSString, AnyObject>> = msg_send![
                class!(NSDictionary),
                dictionaryWithObjects: &*vals_arr
                forKeys: &*keys_arr
            ];

            let mut prov_error: *mut NSError = std::ptr::null_mut();
            let provider: *mut AnyObject = msg_send![provider_cls, alloc];
            let provider: Retained<AnyObject> = {
                let obj: *mut AnyObject = msg_send![
                    provider,
                    initWithDictionary: &*dict
                    error: &mut prov_error
                ];
                if obj.is_null() {
                    return Err("MLDictionaryFeatureProvider init failed".to_string());
                }
                Retained::retain(obj).unwrap()
            };

            // Run prediction
            let start = Instant::now();
            let mut pred_error: *mut NSError = std::ptr::null_mut();
            let _result: *mut AnyObject = msg_send![
                &*self.model,
                predictionFromFeatures: &*provider
                error: &mut pred_error
            ];
            let elapsed_us = start.elapsed().as_micros() as i64;

            if !pred_error.is_null() {
                let desc: Retained<NSString> = msg_send![&*pred_error, localizedDescription];
                return Err(format!("prediction failed: {desc}"));
            }

            Ok(elapsed_us)
        }
    }

    /// Benchmark N predictions, return average microseconds.
    pub fn bench(&self, input_name: &str, shape: &[usize], n: u32) -> Result<i64, String> {
        // Warmup
        for _ in 0..5 {
            self.predict_once(input_name, shape)?;
        }
        let start = Instant::now();
        for _ in 0..n {
            self.predict_once(input_name, shape)?;
        }
        Ok((start.elapsed().as_micros() / n as u128) as i64)
    }
}

fn shape_to_nsarray(shape: &[usize]) -> Retained<NSArray<AnyObject>> {
    unsafe {
        let ns_numbers: Vec<Retained<AnyObject>> = shape
            .iter()
            .map(|&dim| {
                let num: Retained<AnyObject> = msg_send![
                    class!(NSNumber),
                    numberWithInteger: dim as i64
                ];
                num
            })
            .collect();
        let refs: Vec<&AnyObject> = ns_numbers.iter().map(|r| &**r).collect();
        NSArray::from_slice(&refs)
    }
}
