// Private Metal.framework declarations the guest plugin compiles against.
//
// The A10X driver (AGXMetalA10 on iPadOS 17.7.11, 21H461) subclasses these Metal.framework base classes; the plugin
// subclasses the same ones, so that framework-side behavior (labels, descriptor handling, validation hooks, reflection,
// the device's limits and feature tables) is Apple's code. Both SDKs export the classes; their headers do not declare
// them. Each declaration states the 21H461 superclass and nothing else: UniXNU's
// scripts/audit-target-web-metal-surface.py checks every @interface line against the target's metadata, and the plugin
// checks the running classes against the generated layout before it registers a device (WebMetalGuard.mm).

#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// The private device protocol. Metal's MTLAddDevice asserts that a device conforms to it; the definition the runtime
// uses is Metal.framework's, this one only lets the plugin's classes adopt it.
@protocol MTLDeviceSPI <MTLDevice>
@end

@interface _MTLObjectWithLabel : NSObject
@end

@interface _MTLDevice : NSObject
@end

@interface _MTLCommandQueue : _MTLObjectWithLabel
@end

@interface _MTLCommandBuffer : _MTLObjectWithLabel
@end

@interface _MTLCommandEncoder : _MTLObjectWithLabel
@end

@interface _MTLParallelRenderCommandEncoder : _MTLObjectWithLabel
@end

@interface _MTLDebugCommandEncoder : _MTLCommandEncoder
@end

@interface _MTLAccelerationStructureCommandEncoder : _MTLCommandEncoder
@end

@interface _MTLFence : _MTLObjectWithLabel
@end

@interface _MTLResource : _MTLObjectWithLabel
@end

@interface _MTLHeap : _MTLObjectWithLabel
@end

@interface _MTLBinaryArchive : _MTLObjectWithLabel
@end

@interface _MTLDynamicLibrary : _MTLObjectWithLabel
@end

@interface _MTLIndirectArgumentEncoder : _MTLObjectWithLabel
@end

@interface _MTLRenderPipelineState : NSObject
@end

@interface _MTLComputePipelineState : NSObject
@end

@interface _MTLDepthStencilState : NSObject
@end

@interface _MTLSamplerState : NSObject
@end

@interface _MTLFunctionHandle : NSObject
@end

@interface _MTLIndirectRenderCommand : NSObject
@end

@interface _MTLIndirectComputeCommand : NSObject
@end

@interface _MTLRasterizationRateMap : NSObject
@end
