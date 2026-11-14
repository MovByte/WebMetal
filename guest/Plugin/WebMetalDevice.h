// The Metal device of the WebMetal accelerator service, under the A10X driver's names.
//
// Metal.framework matches services in category IOAcceleratorES, loads the bundle the service names in MetalPluginName
// from /System/Library/Extensions and creates the class named by MetalPluginClassName with -initWithAcceleratorPort:,
// then registers the device with MTLAddDevice. For the WebMetal accelerator those are AGXMetalA10.bundle and
// AGXA10XDevice (wire/README.md, "Transport"), as for the A10X's accelerator on 21H461.
//
// The bundle's Info.plist is 21H461's AGXMetalA10.bundle's but for CFBundleExecutable, which names this executable,
// WebMetal. The 21H461 dyld shared cache holds /System/Library/Extensions/AGXMetalA10.bundle/AGXMetalA10, and dyld
// loads a path the cache holds from the cache whatever is on disk (dyld-1165.3 Loader.cpp), so an executable under the
// driver's name would load the A10X driver itself. The bundle is signed with the resource rules of 21H461's
// _CodeSignature/CodeResources (ResourceRules.plist), which seal nothing outside Resources/: its CodeResources is the
// driver bundle's, and the driver's resource files beside it on the system volume stay unsealed, as they are there.
//
// The device classes keep 21H461's chain inside the bundle, AGXA10XDevice -> AGXA10FamilyDevice, each with the methods
// AGXMetalA10 defines on it. The driver's chain continues through IOGPU.framework (AGXA10FamilyDevice ->
// IOGPUMetalDevice -> _MTLDevice); the plugin's AGXA10FamilyDevice subclasses the same Metal.framework base,
// _MTLDevice, directly and carries IOGPUMetalDevice's methods. The device answers the A10X's capability getters from
// the device profile, and every other selector the A10X driver implements is a LOUD stub until a package implements it
// (guest/Generated/surface.tsv lists them).

#pragma once

#import "../Private/MetalPrivate.h"

#include <IOKit/IOKitLib.h>

// The device's state, and the methods of AGXMetalA10's AGXA10FamilyDevice and IOGPU.framework's IOGPUMetalDevice.
@interface AGXA10FamilyDevice : _MTLDevice <MTLDeviceSPI>
@end

// The class Metal.framework creates, with what 21H461's AGXA10XDevice defines itself: -initWithAcceleratorPort: and
// -familyName.
@interface AGXA10XDevice : AGXA10FamilyDevice

// The accelerator service the device belongs to. The device retains it. Returns nil, after a report, when the running
// Metal.framework is not the one the plugin was built for, the service's IOGPU device cannot be created
// (WebMetalIOGPUDevice.h), or the service has no registry entry ID.
- (instancetype)initWithAcceleratorPort:(io_service_t)service;

@end
