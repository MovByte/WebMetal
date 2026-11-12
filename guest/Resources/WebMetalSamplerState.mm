// Sampler states: WebMetalSamplerState stands in for AGXA10FamilySampler (AGXA10FamilySampler -> _MTLSamplerState on
// 21H461).
//
// The driver encodes the descriptor into the A10X's sampler words (AGX::SamplerStateEncoderGen2) and keeps them in the
// object; the GPU reads them. The plugin's GPU is the host's, which builds its own form of the state from the
// descriptor (CreateSamplerState). What the object answers is Metal.framework's (_MTLSamplerState keeps the device,
// label and pixel format) and the driver's two answers below.

#import "WebMetalResources.h"

// The class adopts MTLSamplerState and MTLSamplerStateSPI as AGXA10FamilySampler does and implements only what the
// 21H461 driver implements; the protocols' other methods are Metal.framework's or LOUD stubs.
#pragma clang diagnostic ignored "-Wprotocol"
#pragma clang diagnostic ignored "-Wobjc-protocol-property-synthesis"

@interface _MTLSamplerState (WebMetalFramework)
- (instancetype)initWithDevice:(id<MTLDevice>)device samplerDescriptor:(MTLSamplerDescriptor *)descriptor;
@end

@interface WebMetalSamplerState : _MTLSamplerState <MTLSamplerStateSPI>
@end

@interface WebMetalSamplerState () {
@public
    WebMetalDevice *_owner;
    uint64_t _handle;
    // AGXA10FamilySampler._impl: the driver's sampler state, whose address is the sampler's unique identifier.
    uint64_t _impl;
}
@end

namespace webmetal {

// -[AGXA10FamilyDevice newSamplerStateWithDescriptor:] (AGXMetalA10 0x21c8d7dc0) builds the driver descriptor and
// calls -[AGXA10FamilySampler initWithDevice:samplerDescriptor:driverDescriptor:initialQOSOverride:] (0x21c97f330),
// which initializes _MTLSamplerState with the device and the descriptor first. Neither refuses a descriptor.
id<MTLSamplerState> NewSamplerState(WebMetalDevice *device, SEL _cmd, MTLSamplerDescriptor *d)
{
    WebMetalSamplerState *sampler = [[WebMetalSamplerState alloc] initWithDevice:device samplerDescriptor:d];
    if (!sampler)
        return nil;
    sampler->_owner = device;
    Connection *connection = DeviceConnection(device);
    sampler->_handle = connection->NewHandle();
    wire::CreateSamplerState request;
    request.sampler = sampler->_handle;
    wire::SamplerDescriptor &w = request.descriptor;
    w.borderColor = d.borderColor;
    w.borderColorSpi = Get<uint64_t>(device, _cmd, d, "borderColorSPI");
    w.compareFunction = d.compareFunction;
    w.customBorderColorValue0 = Get<uint32_t>(device, _cmd, d, "customBorderColorValue_0");
    w.customBorderColorValue1 = Get<uint32_t>(device, _cmd, d, "customBorderColorValue_1");
    w.customBorderColorValue2 = Get<uint32_t>(device, _cmd, d, "customBorderColorValue_2");
    w.customBorderColorValue3 = Get<uint32_t>(device, _cmd, d, "customBorderColorValue_3");
    w.forceResourceIndex = Get<BOOL>(device, _cmd, d, "forceResourceIndex");
    w.forceSeamsOnCubemapFiltering = Get<BOOL>(device, _cmd, d, "forceSeamsOnCubemapFiltering");
    if (!wire::strFromNSString(d.label, w.label))
        WMLoudMethod(device, _cmd, "the label has no UTF-8 form, which the wire format carries");
    w.lodAverage = d.lodAverage;
    w.lodBias = Get<float>(device, _cmd, d, "lodBias");
    w.lodMaxClamp = d.lodMaxClamp;
    w.lodMinClamp = d.lodMinClamp;
    w.magFilter = d.magFilter;
    w.maxAnisotropy = d.maxAnisotropy;
    w.minFilter = d.minFilter;
    w.mipFilter = d.mipFilter;
    w.normalizedCoordinates = d.normalizedCoordinates;
    w.pixelFormat = Get<uint64_t>(device, _cmd, d, "pixelFormat");
    w.rAddressMode = d.rAddressMode;
    w.resourceIndex = Get<uint64_t>(device, _cmd, d, "resourceIndex");
    w.sAddressMode = d.sAddressMode;
    w.supportArgumentBuffers = d.supportArgumentBuffers;
    w.tAddressMode = d.tAddressMode;
    wire::Reply<wire::CreateSamplerStateReply> reply;
    connection->Call(device, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(device, _cmd, "the host answered with a Metal error, which the A10X driver does not return here");
    return sampler;
}

}  // namespace webmetal

@implementation WebMetalSamplerState

- (void)dealloc
{
    if (_handle) {
        webmetal::wire::Release request;
        request.object = _handle;
        webmetal::DeviceConnection(_owner)->CallReportOnly<webmetal::wire::Release, webmetal::wire::ReleaseReply>(
            "-[WebMetalSamplerState dealloc]", request);
    }
}

// -[AGXA10FamilySampler resourceIndex] (AGXMetalA10 0x21c97f4f4): 0.
- (uint64_t)resourceIndex
{
    return 0;
}

// -[AGXA10FamilySampler uniqueIdentifier] (AGXMetalA10 0x21c97f4e4): the address of the sampler's driver state.
- (uint64_t)uniqueIdentifier
{
    return reinterpret_cast<uint64_t>(&_impl);
}

@end
