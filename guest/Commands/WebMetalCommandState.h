// State the command path's classes share (WebMetalCommands.h). Internal to guest/Commands.

#pragma once

#import "WebMetalCommands.h"
#import "../Plugin/WebMetalReport.h"

#include <atomic>
#include <memory>

namespace webmetal {

// The stamps of one committed command buffer's encoders (WebMetalCommandBuffer.mm): its top-level encoders in order,
// how many of them have completed (the host reports them in order with EncoderCompleted), how far the kernel has
// processed it, and whether the whole command buffer has completed. Read and written under the registry's lock.
struct StampCompletion {
    std::vector<uint64_t> segments;
    size_t completed = 0;
    bool done = false;
    // The command-buffer-level event waits it records, and how many of them the host has passed (EventWaitPassed).
    uint32_t waits = 0;
    uint32_t passed = 0;
    // The host has processed all of it (CommandBufferScheduled).
    bool scheduled = false;
    // The earlier command buffers of its queue that record a wait and had not been processed when it was committed:
    // the kernel processes a queue's command buffers in order, so this one only after them.
    std::vector<std::shared_ptr<StampCompletion>> behind;
    // The kernel resources its uses are registered under.
    std::vector<uint64_t> keys;
};

// Registers the resource uses of a stream about to be committed, behind the earlier command buffers `behind` of its
// queue (StampsHoldQueue): the kernel stamps an encoder's listed resources when it processes the encoder.
std::shared_ptr<StampCompletion> RegisterStamps(const Stream &stream,
                                                std::vector<std::shared_ptr<StampCompletion>> behind);
// Whether a registered command buffer may still hold the later ones of its queue: it records an event wait and the
// host has not processed all of it.
bool StampsHoldQueue(const std::shared_ptr<StampCompletion> &completion);
// The host reported encoder `encoder`'s work done, passed the command buffer's first `waits` event waits, processed all
// of it, or completed it.
void EncoderStampsCompleted(const std::shared_ptr<StampCompletion> &completion, uint64_t encoder);
void CommandBufferStampsWaitPassed(const std::shared_ptr<StampCompletion> &completion, uint32_t waits);
void CommandBufferStampsScheduled(const std::shared_ptr<StampCompletion> &completion);
void CommandBufferStampsCompleted(const std::shared_ptr<StampCompletion> &completion);

// One command buffer, from creation to completion.
struct CommandBufferState {
    CommandBufferState(Connection *connection, uint64_t handle, bool retainReferences)
        : connection(std::move(connection)), handle(handle), stream(retainReferences),
          releases(std::make_shared<PendingRelease>())
    {
    }

    Connection *const connection;
    // The command buffer's wire handle; the host binds it at Commit.
    const uint64_t handle;
    Stream stream;
    // Encoder handles of this command buffer, released once the host has bound them and their objects are gone.
    const std::shared_ptr<PendingRelease> releases;
    // The host accepted the Commit and bound the handles.
    std::atomic<bool> bound{false};
    // The stamps its commit registered.
    std::shared_ptr<StampCompletion> stamps;
};

// The encoder state every encoder class keeps: where its records go.
struct EncoderState {
    std::shared_ptr<CommandBufferState> commandBuffer;
    // The stream the encoder appends to: its command buffer's, or for a sub-encoder of a parallel render encoder, its
    // own (spliced into the parent's at the parent's endEncoding).
    std::shared_ptr<Stream> own;
    Stream *stream = nullptr;
    // Where the encoder's handle is released once the host has bound it.
    std::shared_ptr<PendingRelease> releases;
    uint64_t handle = 0;
    bool ended = false;
};

}  // namespace webmetal

@interface WebMetalCommandQueue () {
@public
    id<MTLDevice> _device;
    webmetal::Connection *_connection;
    uint64_t _handle;
    // IOGPUMetalCommandQueue._priority and _backgroundPriority: the priorities the kernel last applied
    // (guest/SPI/WebMetalQueueSPI.mm).
    uint64_t _gpuPriority;
    uint64_t _backgroundGPUPriority;
}
@end

@interface WebMetalCommandBuffer () {
@public
    std::shared_ptr<webmetal::CommandBufferState> _state;
    // The encoder Metal.framework's base records as current (-setCurrentCommandEncoder:), kept here because no
    // framework method reads it back.
    __unsafe_unretained id _currentEncoder;
    uint64_t _protectionOptions;
}
@end

@interface WebMetalRenderCommandEncoder () {
@public
    std::shared_ptr<webmetal::EncoderState> _encoder;
}
@end

@interface WebMetalComputeCommandEncoder () {
@public
    std::shared_ptr<webmetal::EncoderState> _encoder;
    MTLDispatchType _dispatchType;
    uint64_t _markerID;
}
@end

@interface WebMetalBlitCommandEncoder () {
@public
    std::shared_ptr<webmetal::EncoderState> _encoder;
}
@end

@interface WebMetalParallelRenderCommandEncoder () {
@public
    std::shared_ptr<webmetal::EncoderState> _encoder;
    // The sub-encoders' streams, in creation order, spliced into the command buffer's stream at endEncoding.
    std::vector<std::shared_ptr<webmetal::Stream>> _subStreams;
    std::mutex _subStreamsLock;
    MTLRenderPassDescriptor *_descriptor;
}
@end

namespace webmetal {

// Records a command-buffer-level record (outside any encoder). Fails the call LOUD when the record cannot be encoded.
template <class Record> void EmitCommandBufferRecord(WebMetalCommandBuffer *buffer, SEL cmd, const Record &record);

// The handle of an object an argument names, noted in `stream`; 0 for nil. An object that has no WebMetal handle fails
// the call LOUD: the plugin cannot name it to the host.
uint64_t Reference(Stream &stream, id self, SEL cmd, id object);

// An object an encoder command names, noted in the encoder's stream. nil where the record needs an object is LOUD.
inline uint64_t EncoderReference(id self, SEL cmd, EncoderState &encoder, id object, bool nilAllowed)
{
    if (!object && !nilAllowed)
        WMLoudMethod(self, cmd, "a nil argument where the command needs an object; Metal leaves it undefined");
    return Reference(*encoder.stream, self, cmd, object);
}

// A new encoder of `buffer` that records into the buffer's stream, or into `subStream` (a parallel sub-encoder).
std::shared_ptr<EncoderState> NewEncoderState(WebMetalCommandBuffer *buffer, Stream *subStream);

// Appends an encoder's record. Fails the call LOUD when the encoder has ended or the record cannot be encoded.
template <class Record> void EmitEncoderRecord(id self, SEL cmd, EncoderState &encoder, const Record &record)
{
    if (encoder.ended)
        WMLoudMethod(self, cmd, "the encoder has ended encoding; Metal leaves later commands undefined");
    if (!wire::encode(encoder.stream->writer(), record))
        WMLoudMethod(self, cmd, "the arguments break a rule of the wire format");
}

// The NSError a failed command buffer completes with, from the host's failure. With `report`, statuses 2 to 4 are
// reported through the not-implemented channel first.
NSError *CommandBufferError(const wire::Failure &failure, const char *what, bool report = true);

}  // namespace webmetal
