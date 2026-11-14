// AGXPrincipalDevice: the bundle's principal class (Info.plist NSPrincipalClass), as AGXMetalA10's on 21H461.
//
// Metal.framework creates the class the accelerator service names in MetalPluginClassName and takes the bundle's
// principal class only for a service that names none (_getMetalPluginClassForService). The A10X's accelerator names
// AGXA10XDevice (the AGX kext sets the property at kernelcache 0xfffffff005c28808), so this class creates no device:
// -[AGXPrincipalDevice initWithAcceleratorPort:] (AGXMetalA10 0x21c8d1adc) writes that loading the bundle through it is
// not supported to standard error, logs the line, prefixed "AGX: ", to the default log at the error and then the info
// level, and answers nil.

#import <Foundation/Foundation.h>

#include <IOKit/IOKitLib.h>
#include <os/log.h>

#include <cstdio>

@interface AGXPrincipalDevice : NSObject
// The 21H461 initializer does not release its receiver when it answers nil; outside the init family, ARC does not
// either.
- (id)initWithAcceleratorPort:(io_service_t)service __attribute__((objc_method_family(none)));
@end

@implementation AGXPrincipalDevice

- (id)initWithAcceleratorPort:(io_service_t)service
{
    fputs("Attempting to load Metal bundle via AGXPrincipalDevice is not supported!\n", stderr);
    os_log_error(OS_LOG_DEFAULT, "AGX: Attempting to load Metal bundle via AGXPrincipalDevice is not supported!\n");
    os_log_info(OS_LOG_DEFAULT, "AGX: Attempting to load Metal bundle via AGXPrincipalDevice is not supported!\n");
    return nil;
}

@end
