// Driver SPI that the system's clients use (QuartzCore, CoreImage, CoreVideo, Metal Performance Shaders,
// AppleMetalGLRenderer, WebKit).
//
// The methods are categories on the plugin classes that stand in for the 21H461 driver classes (guest/Plugin/roles.toml).
// Each comments the 21H461 method it reproduces, by symbol and address in the 21H461 dyld shared cache, or the 21H461
// kernel function where the driver asks the kernel.

#pragma once

#import "../Commands/WebMetalCommands.h"
#import "../Plugin/WebMetalDevice.h"
#import "../Plugin/WebMetalReport.h"
#import "../Resources/WebMetalResources.h"
#import "../Shaders/WebMetalShaders.h"

namespace webmetal {

// MGGetBoolAnswer(@"InternalBuild") (libMobileGestalt), which -[AGXA10FamilyDevice
// initWithAcceleratorPort:simultaneousInstances:] keeps in the device (AGX::Device +0x905, at 0x21c8d32c8) and the
// driver's internal-build paths test. Read once; fails the call of `self` and `_cmd` LOUD when this process has no
// MGGetBoolAnswer.
bool InternalBuild(id self, SEL _cmd);

// Whether the process's entitlements give `name` the boolean true, as IOGPUFamily tests them for the task
// (0xfffffff005bcf080 in the 21H461 kernel: the value of the entitlements dictionary is kOSBooleanTrue). Read once per
// name.
bool TaskHasEntitlement(const char *name);

}  // namespace webmetal

// The command queue's GPU priorities (WebMetalQueueSPI.mm).
@protocol WebMetalQueuePriority
- (BOOL)setGPUPriority:(NSUInteger)priority;
@end
