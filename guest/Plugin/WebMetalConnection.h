// The device's connection to the WebMetal accelerator service (wire/README.md, "Transport" and "Channels").
//
// One connection per device. It allocates the object handles of the device's objects (unique for the life of the
// connection, never reused, never 0) and runs control requests: each request record is written into the caller's
// memory and handed to the service's control selector, which answers with one reply record.
//
// A reply's status decides what the caller does (wire/README.md, "Control"): 0 is success; 1 is a failure Metal
// defines, which the caller turns into Metal's own result (nil, an NSError); 2 and 3 are things the host does not
// implement or cannot honor and 4 a protocol violation, which are reported through the not-implemented channel and fail
// the call. Nothing is answered in place of a reply.

#pragma once

#import <Foundation/Foundation.h>

#include "../../wire/generated/WebMetalWire.h"

#include <IOKit/IOKitLib.h>
#include <dispatch/dispatch.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace webmetal {

// Carries one control request to the host and returns its reply record. Returns false and describes the problem when
// the request could not be delivered or answered.
using Transport = bool (*)(void *context, const uint8_t *request, size_t length, std::vector<uint8_t> *reply,
                           std::string *problem);

// A transport that takes the place of the accelerator service's user client, for a process with no such service (the
// macOS test shim). Set before the first device is created; connections opened afterwards use it.
void SetTransportOverride(Transport transport, void *context);

class Connection {
public:
    // Opens the service's user client (type 0) and exchanges the schema hash (Open). Returns null and describes the
    // problem when the service cannot be opened or refuses the connection.
    static std::unique_ptr<Connection> Open(io_service_t service, std::string *problem);
    ~Connection();

    Connection(const Connection &) = delete;
    Connection &operator=(const Connection &) = delete;

    // A fresh object handle.
    uint64_t NewHandle() { return next_.fetch_add(1, std::memory_order_relaxed); }

    // Runs `request` and decodes its reply into `reply`. Returns true when a reply arrived with status 0 or 1 (a
    // failure Metal defines, left in reply->failure for the caller). Any other outcome is reported for `self` and
    // `_cmd` and raises WebMetalNotImplemented; under strict mode the report aborts.
    template <class Request, class ReplyFields>
    bool Call(id self, SEL _cmd, const Request &request, wire::Reply<ReplyFields> *reply)
    {
        wire::Writer writer;
        if (!wire::encode(writer, request))
            Fail(self, _cmd, Request::kOpcode, "the request breaks a rule of the wire format");
        std::vector<uint8_t> bytes;
        std::string problem;
        if (!Exchange(writer.out, &bytes, &problem))
            Fail(self, _cmd, Request::kOpcode, problem.c_str());
        wire::DecodeError error;
        if (!wire::decodeReply(bytes.data(), bytes.size(), *reply, &error))
            Fail(self, _cmd, Request::kOpcode, "the reply does not decode");
        if (!reply->ok && reply->failure.status != 1)
            Refused(self, _cmd, Request::kOpcode, reply->failure);
        return true;
    }

    // Runs `request` and decodes its reply into `reply` without reporting or raising: returns false, with the reason,
    // when the request cannot be encoded or delivered or the reply does not decode. A reply with a failure status
    // returns true with reply->ok false. For callers that report and fail in their own way (a dispatch queue's block).
    template <class Request, class ReplyFields>
    bool TryCall(const Request &request, wire::Reply<ReplyFields> *reply, std::string *problem)
    {
        wire::Writer writer;
        if (!wire::encode(writer, request)) {
            *problem = "the request breaks a rule of the wire format";
            return false;
        }
        std::vector<uint8_t> bytes;
        if (!Exchange(writer.out, &bytes, problem))
            return false;
        wire::DecodeError error;
        if (!wire::decodeReply(bytes.data(), bytes.size(), *reply, &error)) {
            *problem = "the reply does not decode";
            return false;
        }
        return true;
    }

    // Events (wire/README.md, "Events"). Asks the host to ring `doorbell` on `queue` whenever events for the command
    // queue `queueHandle` (0: device-level events) become pending; the doorbell collects them with FetchEvents. On the
    // target the user client's register_event_port selector delivers an async completion to a notification port that
    // dispatches on `queue`. Returns false, with the reason, when the host refuses.
    bool Watch(uint64_t queueHandle, dispatch_queue_t queue, dispatch_block_t doorbell, std::string *problem);
    // Stops the doorbell of `queueHandle`; none runs once this returns, except one already queued on its queue.
    void Unwatch(uint64_t queueHandle);
    // Rings the doorbell of `queueHandle` on every open connection (the macOS test shim's executor calls this when it
    // posts events, in place of the async completion). It only queues work and may be called from any thread.
    static void Ring(uint64_t queueHandle);

    // As Call, for a request sent where nothing can be raised (a -dealloc): failures are reported only. Returns whether
    // the request succeeded.
    template <class Request, class ReplyFields>
    bool CallReportOnly(const char *what, const Request &request)
    {
        wire::Writer writer;
        std::vector<uint8_t> bytes;
        std::string problem;
        wire::Reply<ReplyFields> reply;
        wire::DecodeError error;
        if (!wire::encode(writer, request))
            problem = "the request breaks a rule of the wire format";
        else if (!Exchange(writer.out, &bytes, &problem))
            ;
        else if (!wire::decodeReply(bytes.data(), bytes.size(), reply, &error))
            problem = "the reply does not decode";
        else if (!reply.ok)
            problem = Describe(reply.failure);
        else
            return true;
        ReportOnly(what, Request::kOpcode, problem);
        return false;
    }

private:
    Connection(io_connect_t connect, Transport transport, void *context);

    bool Exchange(const std::vector<uint8_t> &request, std::vector<uint8_t> *reply, std::string *problem);
    static std::string Describe(const wire::Failure &failure);

    struct Doorbell {
        dispatch_queue_t queue;
        dispatch_block_t block;
        IONotificationPortRef port;  // the async completion's port (none when a transport override is set)
    };
    static void Rang(void *refcon, IOReturn result, void **args, uint32_t count);
    [[noreturn]] static void Fail(id self, SEL _cmd, uint16_t opcode, const char *problem);
    [[noreturn]] static void Refused(id self, SEL _cmd, uint16_t opcode, const wire::Failure &failure);
    static void ReportOnly(const char *what, uint16_t opcode, const std::string &problem);

    io_connect_t connect_;
    Transport transport_;
    void *context_;
    std::atomic<uint64_t> next_{1};
    std::mutex lock_;                // one control call and its fetch_reply at a time
    std::vector<uint8_t> replyBuffer_;
    std::mutex doorbellsLock_;
    std::map<uint64_t, std::unique_ptr<Doorbell>> doorbells_;
};

}  // namespace webmetal
