// Wire handles, Release, events and the clock of the command path (WebMetalCommands.h).

#import "WebMetalCommands.h"

#include <mach/mach_time.h>

#include <unordered_map>

namespace webmetal {

namespace {

std::mutex gHandlesLock;
std::unordered_map<const void *, uint64_t> gHandles;

}  // namespace

void RegisterHandle(id object, uint64_t handle)
{
    std::lock_guard<std::mutex> guard(gHandlesLock);
    gHandles[(__bridge const void *)object] = handle;
}

void UnregisterHandle(id object)
{
    ForgetHandle((__bridge const void *)object);
}

void ForgetHandle(const void *object)
{
    std::lock_guard<std::mutex> guard(gHandlesLock);
    gHandles.erase(object);
}

uint64_t HandleOf(id object)
{
    if (!object)
        return 0;
    {
        std::lock_guard<std::mutex> guard(gHandlesLock);
        auto found = gHandles.find((__bridge const void *)object);
        if (found != gHandles.end())
            return found->second;
    }
    if (ResourceState *state = StateOf(object))
        return state->created ? state->handle : 0;
    return 0;
}

void ReleaseHandle(Connection *connection, uint64_t handle)
{
    wire::Release request;
    request.object = handle;
    connection->CallReportOnly<wire::Release, wire::ReleaseReply>("Release", request);
}

bool FetchEvents(Connection *connection, uint64_t queue, std::vector<wire::Event> &events, std::string *problem)
{
    wire::FetchEvents request;
    request.queue = queue;
    wire::Reply<wire::FetchEventsReply> reply;
    if (!connection->TryCall(request, &reply, problem))
        return false;
    if (!reply.ok) {
        *problem = "the host refused FetchEvents (status " + std::to_string(reply.failure.status) + "): " +
                   reply.failure.message;
        return false;
    }
    const std::vector<uint8_t> &bytes = reply.value.events;
    size_t at = 0;
    while (at < bytes.size()) {
        wire::Event event;
        size_t consumed = 0;
        wire::DecodeError error;
        if (!wire::decodeEvent(bytes.data() + at, bytes.size() - at, event, &consumed, &error)) {
            *problem = "malformed event record at offset " + std::to_string(at + error.offset);
            return false;
        }
        events.push_back(std::move(event));
        at += consumed;
    }
    return true;
}

uint64_t MachTicksFromNanoseconds(uint64_t nanoseconds)
{
    static const mach_timebase_info_data_t timebase = [] {
        mach_timebase_info_data_t info = {};
        mach_timebase_info(&info);
        return info;
    }();
    return (uint64_t)(((__uint128_t)nanoseconds * timebase.denom) / timebase.numer);
}

}  // namespace webmetal
