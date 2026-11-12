// The device's connection to the WebMetal accelerator service (WebMetalConnection.h).

#import "WebMetalConnection.h"
#import "WebMetalReport.h"

#include <mach/mach.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace webmetal {

namespace {

// The service's user-client type and selectors (wire/README.md, "Transport").
constexpr uint32_t kUserClientType = 0;
constexpr uint32_t kSelectorControl = 0;
constexpr uint32_t kSelectorFetchReply = 1;
constexpr uint32_t kSelectorRegisterEventPort = 2;
constexpr size_t kInitialReplyCapacity = 4096;

Transport gOverride;
void *gOverrideContext;

// The open connections, for Connection::Ring.
std::mutex gConnectionsLock;
std::vector<Connection *> gConnections;

std::string opcodeName(uint16_t opcode)
{
    char text[16];
    snprintf(text, sizeof text, "0x%04x", opcode);
    return text;
}

}  // namespace

void SetTransportOverride(Transport transport, void *context)
{
    gOverride = transport;
    gOverrideContext = context;
}

Connection::Connection(io_connect_t connect, Transport transport, void *context)
    : connect_(connect), transport_(transport), context_(context), replyBuffer_(kInitialReplyCapacity)
{
    std::lock_guard<std::mutex> guard(gConnectionsLock);
    gConnections.push_back(this);
}

Connection::~Connection()
{
    {
        std::lock_guard<std::mutex> guard(gConnectionsLock);
        gConnections.erase(std::find(gConnections.begin(), gConnections.end(), this));
    }
    {
        std::lock_guard<std::mutex> guard(doorbellsLock_);
        for (auto &entry : doorbells_)
            if (entry.second->port)
                IONotificationPortDestroy(entry.second->port);
        doorbells_.clear();
    }
    if (connect_)
        IOServiceClose(connect_);
}

bool Connection::Watch(uint64_t queueHandle, dispatch_queue_t queue, dispatch_block_t doorbell, std::string *problem)
{
    auto entry = std::make_unique<Doorbell>(Doorbell{queue, doorbell, nullptr});
    if (!transport_) {
        entry->port = IONotificationPortCreate(kIOMainPortDefault);
        if (!entry->port) {
            *problem = "IONotificationPortCreate failed";
            return false;
        }
        IONotificationPortSetDispatchQueue(entry->port, queue);
        uint64_t reference[kOSAsyncRef64Count] = {};
        reference[kIOAsyncCalloutFuncIndex] = reinterpret_cast<uint64_t>(&Rang);
        reference[kIOAsyncCalloutRefconIndex] = reinterpret_cast<uint64_t>(entry.get());
        uint64_t in[1] = {queueHandle};
        const kern_return_t status =
            IOConnectCallAsyncScalarMethod(connect_, kSelectorRegisterEventPort, IONotificationPortGetMachPort(entry->port),
                                           reference, kOSAsyncRef64Count, in, 1, nullptr, nullptr);
        if (status != KERN_SUCCESS) {
            IONotificationPortDestroy(entry->port);
            char text[80];
            snprintf(text, sizeof text, "register_event_port failed (status 0x%x)", status);
            *problem = text;
            return false;
        }
    }
    std::lock_guard<std::mutex> guard(doorbellsLock_);
    auto &slot = doorbells_[queueHandle];
    if (slot && slot->port)
        IONotificationPortDestroy(slot->port);
    slot = std::move(entry);
    return true;
}

void Connection::Unwatch(uint64_t queueHandle)
{
    std::lock_guard<std::mutex> guard(doorbellsLock_);
    auto found = doorbells_.find(queueHandle);
    if (found == doorbells_.end())
        return;
    if (found->second->port)
        IONotificationPortDestroy(found->second->port);
    doorbells_.erase(found);
}

void Connection::Ring(uint64_t queueHandle)
{
    std::lock_guard<std::mutex> guard(gConnectionsLock);
    for (Connection *connection : gConnections) {
        std::lock_guard<std::mutex> doorbells(connection->doorbellsLock_);
        auto found = connection->doorbells_.find(queueHandle);
        if (found != connection->doorbells_.end())
            dispatch_async(found->second->queue, found->second->block);
    }
}

// IOAsyncCallback of register_event_port: runs on the doorbell's queue.
void Connection::Rang(void *refcon, IOReturn result, void **, uint32_t)
{
    Doorbell *entry = static_cast<Doorbell *>(refcon);
    if (result != kIOReturnSuccess) {
        char text[96];
        snprintf(text, sizeof text, "the service completed the event port with status 0x%x", result);
        WMReport("register_event_port", text);
        return;
    }
    entry->block();
}

std::unique_ptr<Connection> Connection::Open(io_service_t service, std::string *problem)
{
    io_connect_t connect = 0;
    if (!gOverride) {
        const kern_return_t status = IOServiceOpen(service, mach_task_self(), kUserClientType, &connect);
        if (status != KERN_SUCCESS) {
            char text[96];
            snprintf(text, sizeof text, "IOServiceOpen of user-client type %u failed (status 0x%x)", kUserClientType,
                     status);
            *problem = text;
            return nullptr;
        }
    }
    std::unique_ptr<Connection> connection(new Connection(connect, gOverride, gOverrideContext));
    wire::Open open;
    std::copy(wire::kSchemaHash.begin(), wire::kSchemaHash.end(), open.schemaHash.bytes.begin());
    open.pid = getpid();
    wire::Writer writer;
    if (!wire::encode(writer, open)) {
        *problem = "the Open request does not encode";
        return nullptr;
    }
    std::vector<uint8_t> bytes;
    if (!connection->Exchange(writer.out, &bytes, problem))
        return nullptr;
    wire::Reply<wire::OpenReply> reply;
    if (!wire::decodeReply(bytes.data(), bytes.size(), reply)) {
        *problem = "the Open reply does not decode";
        return nullptr;
    }
    if (!reply.ok) {
        *problem = "the service refused the connection: " + Describe(reply.failure);
        return nullptr;
    }
    if (!std::equal(reply.value.schemaHash.bytes.begin(), reply.value.schemaHash.bytes.end(),
                    wire::kSchemaHash.begin())) {
        *problem = "the service answered Open with another schema hash";
        return nullptr;
    }
    return connection;
}

bool Connection::Exchange(const std::vector<uint8_t> &request, std::vector<uint8_t> *reply, std::string *problem)
{
    if (transport_)
        return transport_(context_, request.data(), request.size(), reply, problem);
    std::lock_guard<std::mutex> guard(lock_);
    uint64_t in[4] = {reinterpret_cast<uint64_t>(request.data()), request.size(),
                      reinterpret_cast<uint64_t>(replyBuffer_.data()), replyBuffer_.size()};
    uint64_t length = 0;
    uint32_t count = 1;
    kern_return_t status = IOConnectCallScalarMethod(connect_, kSelectorControl, in, 4, &length, &count);
    if (status == KERN_SUCCESS && length > replyBuffer_.size()) {
        // The reply did not fit: it stays pending until fetch_reply collects it.
        replyBuffer_.resize(length);
        uint64_t fetch[2] = {reinterpret_cast<uint64_t>(replyBuffer_.data()), replyBuffer_.size()};
        count = 1;
        status = IOConnectCallScalarMethod(connect_, kSelectorFetchReply, fetch, 2, &length, &count);
    }
    if (status != KERN_SUCCESS || count != 1 || length > replyBuffer_.size()) {
        char text[96];
        snprintf(text, sizeof text, "the control call failed (status 0x%x, reply length %llu)", status,
                 static_cast<unsigned long long>(length));
        *problem = text;
        return false;
    }
    reply->assign(replyBuffer_.begin(), replyBuffer_.begin() + static_cast<ptrdiff_t>(length));
    return true;
}

std::string Connection::Describe(const wire::Failure &failure)
{
    std::string text;
    switch (failure.status) {
    case 1: text = "Metal error " + failure.domain.value_or("") + " " + std::to_string(failure.code); break;
    case 2: text = "not implemented by the host"; break;
    case 3: text = "the host cannot honor a capability the device advertises"; break;
    case 4: text = "the host refused the request as invalid"; break;
    default: text = "status " + std::to_string(failure.status); break;
    }
    if (!failure.message.empty())
        text += ": " + failure.message;
    return text;
}

void Connection::Fail(id self, SEL _cmd, uint16_t opcode, const char *problem)
{
    const std::string detail = "request " + opcodeName(opcode) + ": " + problem;
    WMLoudMethod(self, _cmd, detail.c_str());
}

void Connection::Refused(id self, SEL _cmd, uint16_t opcode, const wire::Failure &failure)
{
    Fail(self, _cmd, opcode, Describe(failure).c_str());
}

void Connection::ReportOnly(const char *what, uint16_t opcode, const std::string &problem)
{
    WMReport(what, ("request " + opcodeName(opcode) + ": " + problem).c_str());
}

}  // namespace webmetal
