#!/usr/bin/env python3
"""List private ANE selectors registered at runtime for resource-release research; invokes none."""
import ctypes
import json
import sys

if sys.platform != "darwin":
    raise SystemExit("macOS only")
ctypes.CDLL("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine")
objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
objc.objc_getClass.argtypes = [ctypes.c_char_p]
objc.objc_getClass.restype = ctypes.c_void_p
objc.object_getClass.argtypes = [ctypes.c_void_p]
objc.object_getClass.restype = ctypes.c_void_p
objc.class_copyMethodList.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
objc.class_copyMethodList.restype = ctypes.POINTER(ctypes.c_void_p)
objc.method_getName.argtypes = [ctypes.c_void_p]
objc.method_getName.restype = ctypes.c_void_p
objc.sel_getName.argtypes = [ctypes.c_void_p]
objc.sel_getName.restype = ctypes.c_char_p
objc.method_getTypeEncoding.argtypes = [ctypes.c_void_p]
objc.method_getTypeEncoding.restype = ctypes.c_char_p
libc = ctypes.CDLL(None)
libc.free.argtypes = [ctypes.c_void_p]
report = {"classification": "development", "classes": {}}
for name in ["_ANEClient", "_ANEInMemoryModel", "_ANEInMemoryModelDescriptor"]:
    cls = objc.objc_getClass(name.encode())
    if not cls:
        report["classes"][name] = {"missing": True}
        continue
    members = {}
    for kind, owner in [("instance", cls), ("class", objc.object_getClass(cls))]:
        count = ctypes.c_uint()
        methods = objc.class_copyMethodList(owner, ctypes.byref(count))
        members[kind] = sorted([{ "selector": objc.sel_getName(objc.method_getName(methods[i])).decode(), "encoding": objc.method_getTypeEncoding(methods[i]).decode()} for i in range(count.value)], key=lambda method: method["selector"])
        libc.free(methods)
    report["classes"][name] = members
print(json.dumps(report, indent=2))
