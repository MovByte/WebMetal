// Shaders, pipelines, reflection, argument encoders and binary archives, standing in for the AGXMetalA10 classes of
// iPadOS 17.7.11 (21H461) that guest/Plugin/roles.toml names.
//
// Libraries and functions are Metal.framework's own objects (_MTLLibrary, _MTLFunction): the 21H461 driver's library
// methods only call _MTLDevice's, which parses metallibs and asks MTLCompilerService (over XPC) to compile source and to
// specialize functions. The driver's work starts at pipeline creation, where AGXMetalA10 hands each function's AIR
// bitcode to its back end (AGXCompilerCore, also in MTLCompilerService) and builds pipeline objects from the machine
// code. The plugin hands the same bitcode to the host instead (wire/README.md): RegisterAir once per module and
// connection, RegisterFunction once per function object, then the pipeline-creation record with the whole descriptor.
// The host compiles the AIR to WGSL and replies with the pipeline's reflection, from which the plugin builds Metal's
// reflection objects with Metal.framework's own classes.
//
// Each method comments the 21H461 method it reproduces, by symbol and address in the 21H461 dyld shared cache.

#pragma once

#import "../Private/MetalPrivate.h"
#import "../Plugin/WebMetalConnection.h"
#import "../Plugin/WebMetalDevice.h"
#import "../Plugin/WebMetalReport.h"
#import "../Resources/WebMetalResources.h"

#import <Metal/Metal.h>

#include <cstdint>
#include <string>
#include <vector>

// Private protocols the driver classes adopt on 21H461. Metal.framework defines them; these declarations only let the
// plugin's classes adopt them.
@protocol MTLPipelineStateWithProgramAddressTable
@end
@protocol MTLComputePipelineStateSPI <MTLComputePipelineState, MTLPipelineStateWithProgramAddressTable>
@end
@protocol MTLRenderPipelineStateSPI <MTLRenderPipelineState, MTLPipelineStateWithProgramAddressTable>
@end

// Metal.framework methods of the bases and descriptors the shader path calls. 21H461 implements each of them in the
// class named; webmetal::CheckShaderEntryPoints verifies them in the running framework before the first use.
@interface _MTLComputePipelineState (WebMetalFramework)
- (instancetype)initWithDevice:(id<MTLDevice>)device pipelineStateDescriptor:(MTLComputePipelineDescriptor *)descriptor;
@end

@interface _MTLRenderPipelineState (WebMetalFramework)
- (instancetype)initWithDevice:(id<MTLDevice>)device pipelineStateDescriptor:(MTLRenderPipelineDescriptor *)descriptor;
@end

// -[_MTLFunction bitcodeData] (Metal.framework's function class, which no header declares).
@protocol WebMetalFunctionBitcode
- (NSData *)bitcodeData;
@end

@interface _MTLDevice (WebMetalShaders)
- (id<MTLLibrary>)newDefaultLibrary;
- (id<MTLLibrary>)newLibraryWithFile:(NSString *)path error:(NSError **)error;
- (id<MTLLibrary>)newLibraryWithSource:(NSString *)source options:(MTLCompileOptions *)options error:(NSError **)error;
- (void)newLibraryWithSource:(NSString *)source
                     options:(MTLCompileOptions *)options
           completionHandler:(MTLNewLibraryCompletionHandler)completionHandler;
- (id<MTLLibrary>)newLibraryWithData:(dispatch_data_t)data error:(NSError **)error;
- (NSArray *)loadDynamicLibrariesForFunction:(id<MTLFunction>)function
                             insertLibraries:(NSArray *)libraries
                                       error:(NSError **)error;
- (BOOL)validateDynamicLibrary:(id<MTLLibrary>)library state:(BOOL)state error:(NSError **)error;
- (BOOL)validateDynamicLibraryURL:(NSURL *)url error:(NSError **)error;
- (BOOL)validateDynamicLibraryDescriptor:(id)descriptor error:(NSError **)error;
@end

// The pipeline state objects. Each holds the wire handle the host created it under, which the encoders reference.
@interface WebMetalComputePipelineState : _MTLComputePipelineState <MTLComputePipelineStateSPI>
- (instancetype)initWithDevice:(id<MTLDevice>)device pipelineStateDescriptor:(MTLComputePipelineDescriptor *)descriptor;
@end

@interface WebMetalRenderPipelineState : _MTLRenderPipelineState <MTLRenderPipelineStateSPI>
- (instancetype)initWithDevice:(id<MTLDevice>)device pipelineStateDescriptor:(MTLRenderPipelineDescriptor *)descriptor;
@end

@interface WebMetalComputePipelineState () {
@public
    webmetal::Connection *_connection;
    uint64_t _handle;
}
@end

@interface WebMetalRenderPipelineState () {
@public
    webmetal::Connection *_connection;
    uint64_t _handle;
}
@end


namespace webmetal {

// The NSError domain of the A10X driver's own errors: the bundle's name (the CFString "AGXMetalA10" in AGXMetalA10's
// UserCommonShaderFactory, e.g. createComputePipeline at 0x21c8decb4).
extern NSString *const kDriverErrorDomain;

// An NSError in the driver's domain with `message` as its localized description, as the driver makes them
// (+[NSDictionary dictionaryWithObject:forKey:NSLocalizedDescriptionKey], -[NSError initWithDomain:code:userInfo:]).
NSError *DriverError(NSInteger code, NSString *message);

// The NSError of a failure Metal defines (a reply's status 1): its domain, code and message.
NSError *MetalError(const wire::Failure &failure);

// Verifies the Metal.framework methods and instance variables the shader path uses in the running framework, once.
// Returns false and describes the first one missing or with another type encoding.
bool CheckShaderEntryPoints(std::string *problem);
// As CheckShaderEntryPoints, failing the call LOUD for `self` and `_cmd` when the framework lacks one.
void RequireShaderEntryPoints(id self, SEL _cmd);

// The wire handle of a function object (Metal.framework's _MTLFunction) on `device`'s connection: its AIR module is
// registered with the host once per connection (RegisterAir) and the function once per object (RegisterFunction); the
// handle is released when the function is deallocated. A nil function is handle 0. Fails the call LOUD when the
// function has no AIR bitcode or the host refuses it.
uint64_t FunctionHandle(WebMetalDevice *device, id self, SEL _cmd, id<MTLFunction> function);

// The handles of an array of objects that already have wire handles (binary archives, dynamic libraries); an object
// without one fails the call LOUD.
std::vector<uint64_t> ObjectHandles(id self, SEL _cmd, NSArray *objects, const char *what);

// The wire form of a compute or render pipeline descriptor, and the storage its spans and values point into.
struct ComputeDescriptor;
struct RenderDescriptor;

// Metal's reflection objects (Metal.framework's MTLComputePipelineReflectionInternal and
// MTLRenderPipelineReflectionInternal) built from the host's reflection. Fail the call LOUD when the reflection does
// not have the shape the pipeline needs.
MTLComputePipelineReflection *NewComputeReflection(id self, SEL _cmd, const wire::Reflection &reflection);
MTLRenderPipelineReflection *NewRenderReflection(id self, SEL _cmd, const wire::Reflection &reflection,
                                                 MTLVertexDescriptor *vertexDescriptor);

// Gives a pipeline state the wire handle the host created it under: registered for the encoders (guest/Commands,
// RegisterHandle) and released with the object.
void AdoptPipelineHandle(WebMetalComputePipelineState *pipeline, Connection *connection, uint64_t handle);
void AdoptPipelineHandle(WebMetalRenderPipelineState *pipeline, Connection *connection, uint64_t handle);

}  // namespace webmetal
