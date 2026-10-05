//! Development-only release-path probes using selectors enumerated on OS build 26A434.
//! These are experiments, not a promise that private ANE resources are reclaimed.
use objc2::rc::{Allocated, Retained};
use objc2::runtime::{AnyObject, Bool, NSObject};
use objc2::{ClassType, msg_send};
use objc2_foundation::{NSDictionary, NSError, NSString};

use crate::Executable;
use crate::ane_client::ANEClient;

/// Release paths observed in the Objective-C runtime, never guessed selector names.
pub enum ReclaimProbe {
    ModelPurge,
    ClientPurge,
    FreshClientUnload,
    FreshAllocatedClientUnload,
    DropSharedClientReferences,
}

/// Unload a set, try the selected release path, then drop every retained probe object.
/// Normal serving never calls this helper. Purge selectors return void, not confirmation.
pub fn reclaim(executables: Vec<Executable>, probe: ReclaimProbe) -> Vec<String> {
    let models: Vec<_> = executables
        .iter()
        .map(|executable| executable.inner.clone())
        .collect();
    let qos = executables
        .first()
        .map(|executable| executable.qos.0 as u32)
        .unwrap_or(0);
    drop(executables);
    let mut observations = Vec::new();
    match probe {
        ReclaimProbe::ModelPurge => {
            for model in &models {
                unsafe {
                    let _: () = msg_send![&**model, purgeCompiledModel];
                }
            }
            observations.push("_ANEInMemoryModel purgeCompiledModel invoked (void return)".into());
        }
        ReclaimProbe::ClientPurge => {
            if let Some(client) = ANEClient::shared_connection() {
                for model in &models {
                    let compiled: Option<Retained<AnyObject>> =
                        unsafe { msg_send![&**model, model] };
                    if let Some(compiled) = compiled {
                        unsafe {
                            let _: () = msg_send![&*client, purgeCompiledModel: &*compiled];
                        }
                    }
                    if let Some(hash) = model.hex_string_identifier() {
                        unsafe {
                            let _: () = msg_send![&*client, purgeCompiledModelMatchingHash: &*hash];
                        }
                    }
                }
                observations.push("shared _ANEClient purgeCompiledModel: and purgeCompiledModelMatchingHash: invoked (void return)".into());
            }
        }
        ReclaimProbe::FreshClientUnload | ReclaimProbe::FreshAllocatedClientUnload => {
            let client: Option<Retained<ANEClient>> =
                if matches!(probe, ReclaimProbe::FreshAllocatedClientUnload) {
                    let allocated: Allocated<ANEClient> =
                        unsafe { msg_send![ANEClient::class(), alloc] };
                    unsafe { msg_send![allocated, initWithRestrictedAccessAllowed: false] }
                } else {
                    unsafe { msg_send![ANEClient::class(), new] }
                };
            if let Some(client) = client {
                let shared = ANEClient::shared_connection();
                observations.push(format!(
                    "fresh_client_distinct_from_shared={}",
                    shared
                        .as_ref()
                        .is_some_and(|shared| !std::ptr::eq(&**shared, &*client))
                ));
                let options: Retained<NSDictionary<NSString, NSObject>> = NSDictionary::new();
                for model in &models {
                    let compiled: Option<Retained<AnyObject>> =
                        unsafe { msg_send![&**model, model] };
                    if let Some(compiled) = compiled {
                        let mut error: *mut NSError = std::ptr::null_mut();
                        let success: Bool = unsafe {
                            msg_send![&*client, doUnloadModel: &*compiled, options: &*options, qos: qos, error: &mut error]
                        };
                        let description = unsafe { error.as_ref() }
                            .map(|error| error.localizedDescription().to_string());
                        observations.push(format!(
                            "fresh_client_doUnloadModel success={} error={description:?}",
                            success.as_bool()
                        ));
                        let mut error: *mut NSError = std::ptr::null_mut();
                        let success: Bool = unsafe {
                            msg_send![&*client, unloadModel: &*compiled, options: &*options, qos: qos, error: &mut error]
                        };
                        let description = unsafe { error.as_ref() }
                            .map(|error| error.localizedDescription().to_string());
                        observations.push(format!(
                            "fresh_client_unloadModel success={} error={description:?}",
                            success.as_bool()
                        ));
                    }
                }
            } else {
                observations.push("fresh_client_new returned nil".into());
            }
        }
        ReclaimProbe::DropSharedClientReferences => {
            drop(ANEClient::shared_connection());
            observations.push("released returned sharedConnection reference; no reset/invalidate selector was found on _ANEClient".into());
        }
    }
    drop(models);
    observations
}

/// Drain Objective-C autoreleased temporary objects after a development probe.
/// Objects explicitly retained by live executables remain owned across the pool.
pub fn with_autorelease_pool<R>(body: impl FnOnce() -> R) -> R {
    objc2::rc::autoreleasepool(|_| body())
}
