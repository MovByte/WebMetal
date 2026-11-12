// The Metal device of the WebMetal accelerator service.
//
// Metal.framework matches services in category IOAcceleratorES, loads the bundle the service names in MetalPluginName
// from /System/Library/Extensions and creates the class named by MetalPluginClassName with -initWithAcceleratorPort:,
// then registers the device with MTLAddDevice. For the WebMetal accelerator those are WebMetal.bundle and
// WebMetalDevice (wire/README.md, "Transport").
//
// WebMetalDevice stands in for the A10X's AGXA10XDevice: it subclasses the same Metal.framework base (_MTLDevice),
// answers the A10X's capability getters from the device profile, and every other selector the A10X driver implements
// is a LOUD stub until a package implements it (guest/Generated/surface.tsv lists them).

#pragma once

#import "../Private/MetalPrivate.h"

#include <IOKit/IOKitLib.h>

@interface WebMetalDevice : _MTLDevice <MTLDeviceSPI>

// The accelerator service the device belongs to. The device retains it. Returns nil, after a report, when the running
// Metal.framework is not the one the plugin was built for or the service has no registry entry ID.
- (instancetype)initWithAcceleratorPort:(io_service_t)service;

@end
