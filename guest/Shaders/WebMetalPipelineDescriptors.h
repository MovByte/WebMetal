// The wire forms of pipeline descriptors (WebMetalPipelineDescriptors.mm). Internal to guest/Shaders.

#pragma once

#import "WebMetalShaders.h"

#include <deque>

namespace webmetal {

// What a descriptor's spans and value pointers point into. Deques never move their elements, so the pointers stay valid
// while more are added.
struct Storage {
    std::deque<std::vector<uint64_t>> handles;
    std::deque<std::vector<wire::PipelineBufferDescriptor>> buffers;
    std::deque<std::vector<wire::LinkedFunctionGroup>> groups;
    std::deque<std::string> strings;
    std::deque<wire::Value> values;
};

struct ComputeDescriptor {
    ComputeDescriptor() = default;
    ComputeDescriptor(const ComputeDescriptor &) = delete;
    ComputeDescriptor &operator=(const ComputeDescriptor &) = delete;

    wire::ComputePipelineDescriptor descriptor;
    Storage storage;
    std::vector<wire::Attribute> attributes;
    std::vector<wire::BufferLayout> layouts;
};

struct RenderDescriptor {
    RenderDescriptor() = default;
    RenderDescriptor(const RenderDescriptor &) = delete;
    RenderDescriptor &operator=(const RenderDescriptor &) = delete;

    wire::RenderPipelineDescriptor descriptor;
    Storage storage;
    std::vector<wire::RenderPipelineColorAttachment> colors;
    std::vector<wire::VertexAttribute> vertexAttributes;
    std::vector<wire::VertexBufferLayout> vertexLayouts;
};

// Fill `out` from a descriptor. Functions are registered with the host on the way (FunctionHandle). Fail the call LOUD
// when the running framework's descriptor lacks a field the wire carries or an object has no wire form.
void BuildComputeDescriptor(WebMetalDevice *device, id self, SEL _cmd, MTLComputePipelineDescriptor *in,
                            ComputeDescriptor &out);
void BuildRenderDescriptor(WebMetalDevice *device, id self, SEL _cmd, MTLRenderPipelineDescriptor *in,
                           RenderDescriptor &out);

}  // namespace webmetal
