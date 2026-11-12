//! The structure of a committed command-buffer stream.
//!
//! [`plan`] checks a decoded stream against the protocol and turns it into the [`Step`]s the queue replays:
//!
//! - Records follow the placement of their Metal protocol ([`stream_rule`]): command-buffer records outside any
//!   encoder, an encoder's records between its `Begin*` record and its `EncoderEndEncoding`, a parallel render
//!   encoder's own records outside its sub-encoders and each sub-encoder's records between `BeginParallelSubEncoder`
//!   (whose `parent` is the open parallel encoder) and its `EncoderEndEncoding`. A stream that ends with an encoder
//!   open is refused: Metal's driver refuses that commit too (`-[IOGPUMetalCommandBuffer validate]`: "commit command
//!   buffer with uncommitted encoder"), as it refuses an encoder opened while another is open and an event signalled
//!   with an encoder open (measured in the reference cases `command-buffer.encoder-while-encoder-open` and
//!   `event.signal-with-open-encoder`), so the guest never sends such a stream.
//! - Every handle is checked against the task's object table ([`Table::check`]); encoder handles are created by the
//!   stream itself and must be fresh. The Commit's `resources` must list exactly the existing objects the stream names,
//!   each once.
//! - Event and fence records must name an object: Metal raises `NSInvalidArgumentException` for a nil event and
//!   faults on a nil fence at encode time (measured on Apple M4, macOS 27.0.1), so the guest never encodes one.
//!
//! Fence records become no steps. WebGPU runs the passes of a queue in submission order and makes each pass's writes
//! visible to the passes after it, which is everything a legal fence program asks for; Metal evaluates a wait on a
//! fence no earlier pass updated as already reached ("The fence is evaluated at kernel submission"), and a wait after
//! an update in the same encoder, which Metal calls illegal, completes normally on Metal without the debug layer
//! (reference cases `fence.wait-before-later-update`, `fence.update-then-wait-same-encoder`). In-order execution
//! produces each of those outcomes.
//!
//! Records the host does not implement are not protocol errors: the plan records the first one, and the command buffer
//! completes with status 2 and its offset without executing.

use std::collections::{HashMap, HashSet};

use webmetal_wire::{Command, Failure, Handle};

use super::failure;
use super::objects::Table;
use super::schema::{HandleUse, Kind, Opens, Role, Scope, command_handles, stream_rule};

/// The kind of an encoder segment, which selects the service that encodes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EncoderKind {
    Render,
    /// A parallel render encoder with its sub-encoders spliced in, in creation order.
    ParallelRender,
    Compute,
    Blit,
}

/// One encoder's records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub kind: EncoderKind,
    /// The encoder's handle.
    pub encoder: Handle,
    /// Indices into the stream's records, in order, from the `Begin*` record to the closing `EncoderEndEncoding`
    /// (inclusive), without the fence records the core handles.
    pub records: Vec<usize>,
}

/// One step of a command buffer, in stream order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    Encoder(Segment),
    /// `encodeWaitForEvent:value:timeout:` (`encodeWaitForEvent:value:` with [`WAIT_TIMEOUT`]); `timeout` in seconds.
    Wait {
        event: Handle,
        value: u64,
        timeout: u32,
        offset: u64,
    },
    /// `encodeSignalEvent:value:`.
    Signal {
        event: Handle,
        value: u64,
        offset: u64,
    },
    /// `-[MTLCommandBuffer pushDebugGroup:]`.
    PushDebugGroup {
        label: String,
    },
    /// `-[MTLCommandBuffer popDebugGroup]`.
    PopDebugGroup,
}

/// The timeout of `encodeWaitForEvent:value:`, in seconds: 21H461's `-[IOGPUMetalCommandBuffer encodeWaitForEvent:value:]`
/// (0x1fcb328e0) sends `encodeWaitForEvent:value:timeout:` with 5.
pub const WAIT_TIMEOUT: u32 = 5;

/// A stream that obeys the protocol.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub steps: Vec<Step>,
    /// The encoders the stream creates, in creation order: a parallel render encoder's sub-encoders (render encoders
    /// the guest holds and releases like any other) follow it.
    pub encoders: Vec<(Handle, EncoderKind)>,
    /// The first record the host does not implement: its offset and failure.
    pub unimplemented: Option<(u64, Failure)>,
}

/// A protocol violation: the stream offset of the offending record (`None` for the Commit's own fields) and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    pub offset: Option<u64>,
    pub message: String,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.offset {
            Some(o) => write!(f, "stream offset {o}: {}", self.message),
            None => write!(f, "{}", self.message),
        }
    }
}

enum Open {
    Encoder { kind: EncoderKind, handle: Handle, records: Vec<usize> },
    Parallel { handle: Handle, records: Vec<usize>, sub: Option<Handle> },
}

/// Check `records` (the decoded stream) and build its plan. `resources` is the Commit's resource list; `implemented`
/// says whether the executor has a service for an encoder kind.
pub fn plan(
    records: &[(usize, Command)],
    table: &Table,
    resources: &[Handle],
    implemented: &dyn Fn(EncoderKind) -> bool,
) -> Result<Plan, Violation> {
    let mut out = Plan::default();
    let mut open: Option<Open> = None;
    let mut created: HashMap<Handle, EncoderKind> = HashMap::new();
    let mut referenced: HashSet<Handle> = HashSet::new();

    for (index, (offset, record)) in records.iter().enumerate() {
        let offset = *offset as u64;
        let fail = |message: String| Violation { offset: Some(offset), message };
        let opcode = record.opcode();
        let name = webmetal_wire::op_info(opcode).map_or("?", |op| op.name);
        let rule = stream_rule(opcode).ok_or_else(|| fail(format!("{name} is not a stream record")))?;

        // Handles.
        let mut uses: Vec<HandleUse> = Vec::new();
        command_handles(record, &mut |u| uses.push(u));
        for u in &uses {
            match u.role {
                Role::Result => {
                    if created.contains_key(&u.handle) {
                        return Err(fail(format!("{}: handle {:#x} is created twice", u.field, u.handle.get())));
                    }
                    table.check(u).map_err(fail)?;
                }
                _ if created.contains_key(&u.handle) => {
                    if !Kind::Encoder.satisfies(u.kind) {
                        return Err(fail(format!(
                            "{}: handle {:#x} is an Encoder, the field takes a {}",
                            u.field,
                            u.handle.get(),
                            u.kind.name()
                        )));
                    }
                }
                _ => {
                    table.check(u).map_err(fail)?;
                    referenced.insert(u.handle);
                }
            }
        }

        // Placement.
        if let Some(opens) = rule.opens {
            let encoder = uses.iter().find(|u| u.role == Role::Result).map(|u| u.handle).expect("Begin record");
            match opens {
                Opens::ParallelSub => {
                    let parent = match record {
                        Command::BeginParallelSubEncoder(r) => r.parent,
                        _ => unreachable!("only BeginParallelSubEncoder opens a sub-encoder"),
                    };
                    match &mut open {
                        Some(Open::Parallel { handle, records, sub }) if sub.is_none() => {
                            if parent != *handle {
                                return Err(fail(format!(
                                    "BeginParallelSubEncoder: parent {:#x} is not the open parallel encoder {:#x}",
                                    parent.get(),
                                    handle.get()
                                )));
                            }
                            *sub = Some(encoder);
                            records.push(index);
                        }
                        _ => return Err(fail("BeginParallelSubEncoder outside an open parallel encoder".into())),
                    }
                    created.insert(encoder, EncoderKind::Render);
                    out.encoders.push((encoder, EncoderKind::Render));
                }
                _ => {
                    if open.is_some() {
                        return Err(fail(format!(
                            "{name}: a command encoder is already encoding to this command buffer"
                        )));
                    }
                    let kind = match opens {
                        Opens::Render => EncoderKind::Render,
                        Opens::Parallel => EncoderKind::ParallelRender,
                        Opens::Compute => EncoderKind::Compute,
                        Opens::Blit => EncoderKind::Blit,
                        Opens::ParallelSub => unreachable!(),
                    };
                    open = Some(if kind == EncoderKind::ParallelRender {
                        Open::Parallel { handle: encoder, records: vec![index], sub: None }
                    } else {
                        Open::Encoder { kind, handle: encoder, records: vec![index] }
                    });
                    created.insert(encoder, kind);
                    out.encoders.push((encoder, kind));
                }
            }
            continue;
        }

        match rule.scope {
            Scope::CommandBuffer => {
                if open.is_some() {
                    return Err(fail(format!("{name} with an encoder open")));
                }
                command_buffer_record(record, offset, &mut out).map_err(fail)?;
            }
            Scope::AnyEncoder | Scope::Render | Scope::Compute | Scope::Blit | Scope::Parallel => {
                let inside = match (&open, rule.scope) {
                    (None, _) => false,
                    (Some(_), Scope::AnyEncoder) => true,
                    (Some(Open::Encoder { kind, .. }), s) => matches!(
                        (kind, s),
                        (EncoderKind::Render, Scope::Render)
                            | (EncoderKind::Compute, Scope::Compute)
                            | (EncoderKind::Blit, Scope::Blit)
                    ),
                    (Some(Open::Parallel { sub, .. }), Scope::Render) => sub.is_some(),
                    (Some(Open::Parallel { sub, .. }), Scope::Parallel) => sub.is_none(),
                    (Some(Open::Parallel { .. }), _) => false,
                };
                if !inside {
                    return Err(fail(format!("{name} outside an encoder that takes it")));
                }
                if let Some(fence) = fence_record(record) {
                    // Fence records: validated above (a live Fence); no step.
                    if fence.is_none() {
                        return Err(fail(format!("{name}: nil fence")));
                    }
                    continue;
                }
                let end = matches!(record, Command::EncoderEndEncoding(_));
                match open.as_mut().expect("inside an encoder") {
                    Open::Encoder { records, .. } => records.push(index),
                    Open::Parallel { records, sub, .. } => {
                        records.push(index);
                        if end && sub.is_some() {
                            *sub = None;
                            continue;
                        }
                    }
                }
                if end {
                    let segment = match open.take().expect("open encoder") {
                        Open::Encoder { kind, handle, records } => Segment { kind, encoder: handle, records },
                        Open::Parallel { handle, records, .. } => {
                            Segment { kind: EncoderKind::ParallelRender, encoder: handle, records }
                        }
                    };
                    if out.unimplemented.is_none() && !implemented(segment.kind) {
                        let begin = &records[segment.records[0]];
                        out.unimplemented = Some((begin.0 as u64, failure::unimplemented_record(begin.1.opcode())));
                    }
                    out.steps.push(Step::Encoder(segment));
                }
            }
        }
    }

    if let Some(o) = &open {
        let handle = match o {
            Open::Encoder { handle, .. } | Open::Parallel { handle, .. } => *handle,
        };
        return Err(Violation {
            offset: None,
            message: format!(
                "the stream ends with encoder {:#x} open (commit command buffer with uncommitted encoder)",
                handle.get()
            ),
        });
    }

    // The Commit's resource list.
    let mut listed = HashSet::new();
    for &r in resources {
        if !listed.insert(r) {
            return Err(Violation { offset: None, message: format!("resources lists handle {:#x} twice", r.get()) });
        }
    }
    let mut missing: Vec<u64> = referenced.difference(&listed).map(|h| h.get()).collect();
    let mut extra: Vec<u64> = listed.difference(&referenced).map(|h| h.get()).collect();
    missing.sort_unstable();
    extra.sort_unstable();
    if let Some(m) = missing.first() {
        return Err(Violation {
            offset: None,
            message: format!("the stream names handle {m:#x}, which resources does not list"),
        });
    }
    if let Some(e) = extra.first() {
        return Err(Violation {
            offset: None,
            message: format!("resources lists handle {e:#x}, which the stream does not name"),
        });
    }
    Ok(out)
}

/// The fence a fence record names (`Some(None)` for nil), or `None` for any other record.
fn fence_record(record: &Command) -> Option<Option<Handle>> {
    Some(match record {
        Command::RenderUpdateFence(r) => r.fence,
        Command::RenderWaitForFence(r) => r.fence,
        Command::ComputeUpdateFence(r) => r.fence,
        Command::ComputeWaitForFence(r) => r.fence,
        Command::BlitUpdateFence(r) => r.fence,
        Command::BlitWaitForFence(r) => r.fence,
        _ => return None,
    })
}

/// A record placed on the command buffer itself.
fn command_buffer_record(record: &Command, offset: u64, out: &mut Plan) -> Result<(), String> {
    let event = |e: Option<Handle>, what: &str| e.ok_or_else(|| format!("{what}: nil event"));
    match record {
        Command::EncodeSignalEvent(r) => {
            out.steps.push(Step::Signal { event: event(r.event, "EncodeSignalEvent")?, value: r.value, offset });
        }
        Command::EncodeWaitForEvent(r) => {
            let event = event(r.event, "EncodeWaitForEvent")?;
            out.steps.push(Step::Wait { event, value: r.value, timeout: WAIT_TIMEOUT, offset });
        }
        Command::EncodeWaitForEventTimeout(r) => {
            let event = event(r.event, "EncodeWaitForEventTimeout")?;
            out.steps.push(Step::Wait { event, value: r.value, timeout: r.timeout, offset });
        }
        Command::CommandBufferPushDebugGroup(r) => {
            // A nil string is an empty group name; the name is not observable by Metal programs.
            out.steps.push(Step::PushDebugGroup { label: r.string.clone().unwrap_or_default() });
        }
        Command::CommandBufferPopDebugGroup(_) => out.steps.push(Step::PopDebugGroup),
        // The kernel sets the queue's protection options for the rest of the command buffer and returns them to 0
        // after it (IOGPUFamily, kIOGPUKernelCommandSetProtectionOptions): 0 is every command buffer's starting state.
        // What the A10X driver does under other options is not established.
        Command::CommandBufferSetProtectionOptions(r) => {
            if r.options != 0 && out.unimplemented.is_none() {
                out.unimplemented = Some((
                    offset,
                    failure::unimplemented(format!(
                        "-[MTLCommandBufferSPI setProtectionOptions:] {:#x}: not implemented by the host",
                        r.options
                    )),
                ));
            }
        }
        // MTLCommandBufferSPI records whose behavior is not established: the agent mask of a signal, a signal at
        // scheduling time, cache hints and resource groups.
        Command::EncodeSignalEventAgentMask(r) => {
            event(r.event, "EncodeSignalEventAgentMask")?;
            note_unimplemented(out, offset, record);
        }
        Command::EncodeSignalEventScheduled(r) => {
            event(r.event, "EncodeSignalEventScheduled")?;
            note_unimplemented(out, offset, record);
        }
        // Any other command-buffer record (cache hints, resource groups, records a newer schema adds) is not
        // implemented.
        _ => note_unimplemented(out, offset, record),
    }
    Ok(())
}

fn note_unimplemented(out: &mut Plan, offset: u64, record: &Command) {
    if out.unimplemented.is_none() {
        out.unimplemented = Some((offset, failure::unimplemented_record(record.opcode())));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::objects::{Entry, Owner};
    use webmetal_wire::fuzz::Rng;
    use webmetal_wire::*;

    fn h(v: u64) -> Handle {
        Handle::new(v).unwrap()
    }

    /// Encode the records and decode them again, so they carry their real offsets.
    fn stream(records: Vec<Command>) -> Vec<(usize, Command)> {
        let mut bytes = Vec::new();
        for r in &records {
            r.encode(&mut bytes).unwrap();
        }
        Command::decode_all(&bytes).unwrap()
    }

    fn table(objects: &[(u64, Kind)]) -> Table {
        let mut t = Table::default();
        for &(handle, kind) in objects {
            t.bind(h(handle), Entry { kind, owner: Owner::Core, label: None, payload: None }).unwrap();
        }
        t
    }

    fn all(_: EncoderKind) -> bool {
        true
    }

    fn end() -> Command {
        Command::EncoderEndEncoding(EncoderEndEncoding {})
    }

    fn fill(buffer: u64) -> Command {
        Command::BlitFillBuffer(BlitFillBuffer {
            buffer: Some(h(buffer)),
            range: Range { location: 0, length: 4 },
            value: 1,
        })
    }

    fn pass() -> RenderPassDescriptor {
        let mut d = RenderPassDescriptor::random(&mut Rng::new(7));
        d.binary_archives.clear();
        d.color_attachments.clear();
        d.depth_attachment = None;
        d.stencil_attachment = None;
        d.rasterization_rate_map = None;
        d.sample_buffer_attachments.clear();
        d.visibility_result_buffer = None;
        d
    }

    #[test]
    fn encoders_events_and_debug_groups_become_steps() {
        let t = table(&[(1, Kind::Buffer), (2, Kind::Event), (3, Kind::Fence)]);
        let s = stream(vec![
            Command::CommandBufferPushDebugGroup(CommandBufferPushDebugGroup { string: Some("outer".into()) }),
            Command::EncodeWaitForEvent(EncodeWaitForEvent { event: Some(h(2)), value: 1 }),
            Command::BeginBlit(BeginBlit { encoder: h(10) }),
            Command::BlitWaitForFence(BlitWaitForFence { fence: Some(h(3)) }),
            fill(1),
            Command::BlitUpdateFence(BlitUpdateFence { fence: Some(h(3)) }),
            end(),
            Command::EncodeSignalEvent(EncodeSignalEvent { event: Some(h(2)), value: 2 }),
            Command::CommandBufferPopDebugGroup(CommandBufferPopDebugGroup {}),
        ]);
        let p = plan(&s, &t, &[h(2), h(3), h(1)], &all).unwrap();
        assert_eq!(p.encoders, vec![(h(10), EncoderKind::Blit)]);
        assert!(p.unimplemented.is_none());
        assert_eq!(
            p.steps,
            vec![
                Step::PushDebugGroup { label: "outer".into() },
                Step::Wait { event: h(2), value: 1, timeout: WAIT_TIMEOUT, offset: s[1].0 as u64 },
                // The fence records are not part of the segment.
                Step::Encoder(Segment { kind: EncoderKind::Blit, encoder: h(10), records: vec![2, 4, 6] }),
                Step::Signal { event: h(2), value: 2, offset: s[7].0 as u64 },
                Step::PopDebugGroup,
            ]
        );
    }

    #[test]
    fn protection_options_other_than_zero_are_not_implemented() {
        let t = table(&[(1, Kind::Buffer)]);
        let options =
            |options| Command::CommandBufferSetProtectionOptions(CommandBufferSetProtectionOptions { options });
        // Setting 0 again is the kernel's state at the start of every command buffer.
        let s = stream(vec![options(0), Command::BeginBlit(BeginBlit { encoder: h(10) }), fill(1), end()]);
        let p = plan(&s, &t, &[h(1)], &all).unwrap();
        assert!(p.unimplemented.is_none());
        assert_eq!(p.steps.len(), 1);
        let s = stream(vec![Command::BeginBlit(BeginBlit { encoder: h(10) }), fill(1), end(), options(1), options(0)]);
        let p = plan(&s, &t, &[h(1)], &all).unwrap();
        let (offset, f) = p.unimplemented.unwrap();
        assert_eq!(offset, s[3].0 as u64);
        assert_eq!(f.status, webmetal_wire::status::UNIMPLEMENTED);
        assert!(f.message.contains("setProtectionOptions:] 0x1"), "{}", f.message);
        // The record belongs to the command buffer, not to an encoder.
        let s = stream(vec![Command::BeginBlit(BeginBlit { encoder: h(10) }), options(1), end()]);
        assert!(plan(&s, &t, &[], &all).unwrap_err().message.contains("with an encoder open"));
    }

    #[test]
    fn parallel_encoders_carry_their_sub_encoders() {
        let t = table(&[]);
        let s = stream(vec![
            Command::BeginParallelRenderPass(BeginParallelRenderPass { encoder: h(1), render_pass_descriptor: pass() }),
            Command::ParallelSetDepthStoreAction(ParallelSetDepthStoreAction { store_action: 1 }),
            Command::BeginParallelSubEncoder(BeginParallelSubEncoder { parent: h(1), encoder: h(2) }),
            Command::RenderSetCullMode(RenderSetCullMode { cull_mode: 1 }),
            end(),
            Command::BeginParallelSubEncoder(BeginParallelSubEncoder { parent: h(1), encoder: h(3) }),
            Command::EncoderSetLabel(EncoderSetLabel { label: Some("sub".into()) }),
            end(),
            end(),
        ]);
        let p = plan(&s, &t, &[], &all).unwrap();
        assert_eq!(
            p.steps,
            vec![Step::Encoder(Segment {
                kind: EncoderKind::ParallelRender,
                encoder: h(1),
                records: (0..9).collect()
            })]
        );
        assert_eq!(
            p.encoders,
            vec![(h(1), EncoderKind::ParallelRender), (h(2), EncoderKind::Render), (h(3), EncoderKind::Render)]
        );

        // A render record between sub-encoders, or a parallel record inside one, is misplaced.
        let s = stream(vec![
            Command::BeginParallelRenderPass(BeginParallelRenderPass { encoder: h(1), render_pass_descriptor: pass() }),
            Command::RenderSetCullMode(RenderSetCullMode { cull_mode: 1 }),
            end(),
        ]);
        assert!(plan(&s, &t, &[], &all).is_err());
        let s = stream(vec![
            Command::BeginParallelRenderPass(BeginParallelRenderPass { encoder: h(1), render_pass_descriptor: pass() }),
            Command::BeginParallelSubEncoder(BeginParallelSubEncoder { parent: h(1), encoder: h(2) }),
            Command::ParallelSetDepthStoreAction(ParallelSetDepthStoreAction { store_action: 1 }),
            end(),
            end(),
        ]);
        assert!(plan(&s, &t, &[], &all).is_err());
        // The parent must be the open parallel encoder.
        let s = stream(vec![
            Command::BeginParallelRenderPass(BeginParallelRenderPass { encoder: h(1), render_pass_descriptor: pass() }),
            Command::BeginParallelSubEncoder(BeginParallelSubEncoder { parent: h(9), encoder: h(2) }),
            end(),
            end(),
        ]);
        assert!(plan(&s, &t, &[], &all).is_err());
    }

    #[test]
    fn misplaced_records_are_violations() {
        let t = table(&[(1, Kind::Buffer), (2, Kind::Event)]);
        let begin = |e| Command::BeginBlit(BeginBlit { encoder: h(e) });
        // An encoder opened while another is open.
        let s = stream(vec![begin(10), begin(11), end(), end()]);
        let v = plan(&s, &t, &[], &all).unwrap_err();
        assert_eq!(v.offset, Some(s[1].0 as u64));
        // A stream ending with an encoder open.
        let s = stream(vec![begin(10), fill(1)]);
        assert!(plan(&s, &t, &[h(1)], &all).unwrap_err().message.contains("uncommitted encoder"));
        // A signal with an encoder open.
        let s = stream(vec![
            begin(10),
            Command::EncodeSignalEvent(EncodeSignalEvent { event: Some(h(2)), value: 1 }),
            end(),
        ]);
        assert!(plan(&s, &t, &[h(2)], &all).is_err());
        // An encoder record outside an encoder, and a compute record in a blit encoder.
        assert!(plan(&stream(vec![end()]), &t, &[], &all).is_err());
        let s = stream(vec![
            begin(10),
            Command::ComputeSetBuffer(ComputeSetBuffer { buffer: Some(h(1)), offset: 0, index: 0 }),
            end(),
        ]);
        assert!(plan(&s, &t, &[h(1)], &all).is_err());
    }

    #[test]
    fn nil_events_and_fences_are_violations() {
        let t = table(&[]);
        let s = stream(vec![Command::EncodeWaitForEvent(EncodeWaitForEvent { event: None, value: 1 })]);
        assert!(plan(&s, &t, &[], &all).unwrap_err().message.contains("nil event"));
        let s = stream(vec![
            Command::BeginBlit(BeginBlit { encoder: h(10) }),
            Command::BlitUpdateFence(BlitUpdateFence { fence: None }),
            end(),
        ]);
        assert!(plan(&s, &t, &[], &all).unwrap_err().message.contains("nil fence"));
    }

    #[test]
    fn handles_follow_the_object_table() {
        let t = table(&[(1, Kind::Buffer), (2, Kind::Event), (5, Kind::Texture)]);
        let blit = |cmds: Vec<Command>| {
            let mut v = vec![Command::BeginBlit(BeginBlit { encoder: h(10) })];
            v.extend(cmds);
            v.push(end());
            stream(v)
        };
        // Unknown handle, wrong kind.
        assert!(plan(&blit(vec![fill(3)]), &t, &[h(3)], &all).unwrap_err().message.contains("unknown handle"));
        assert!(plan(&blit(vec![fill(5)]), &t, &[h(5)], &all).unwrap_err().message.contains("is a Texture"));
        // An encoder handle already bound in the task, or created twice in one stream.
        assert!(plan(&stream(vec![Command::BeginBlit(BeginBlit { encoder: h(1) }), end()]), &t, &[], &all).is_err());
        let s = stream(vec![
            Command::BeginBlit(BeginBlit { encoder: h(10) }),
            end(),
            Command::BeginBlit(BeginBlit { encoder: h(10) }),
            end(),
        ]);
        assert!(plan(&s, &t, &[], &all).unwrap_err().message.contains("created twice"));
        // resources must list exactly the objects the stream names, once each.
        assert!(plan(&blit(vec![fill(1)]), &t, &[], &all).unwrap_err().message.contains("does not list"));
        assert!(plan(&blit(vec![fill(1)]), &t, &[h(1), h(1)], &all).unwrap_err().message.contains("twice"));
        assert!(plan(&blit(vec![fill(1)]), &t, &[h(1), h(2)], &all).unwrap_err().message.contains("does not name"));
        assert!(plan(&blit(vec![fill(1)]), &t, &[h(1)], &all).is_ok());
    }

    #[test]
    fn unimplemented_records_are_noted_not_refused() {
        let t = table(&[(2, Kind::Event)]);
        let s = stream(vec![
            Command::EncodeSignalEvent(EncodeSignalEvent { event: Some(h(2)), value: 1 }),
            Command::EncodeCacheHintTag(EncodeCacheHintTag { tag: 1, resource_groups: vec![None], count: 1 }),
            Command::EncodeSignalEventAgentMask(EncodeSignalEventAgentMask {
                event: Some(h(2)),
                value: 2,
                agent_mask: 1,
            }),
        ]);
        let p = plan(&s, &t, &[h(2)], &all).unwrap();
        let (offset, f) = p.unimplemented.unwrap();
        assert_eq!(offset, s[1].0 as u64);
        assert_eq!(f.status, status::UNIMPLEMENTED);
        assert!(f.message.contains("encodeCacheHintTag"), "{}", f.message);

        // An encoder without a service is unimplemented at its Begin record.
        let s = stream(vec![Command::BeginCompute(BeginCompute { encoder: h(10) }), end()]);
        let p = plan(&s, &t, &[], &|k| k != EncoderKind::Compute).unwrap();
        let (offset, f) = p.unimplemented.unwrap();
        assert_eq!(offset, 0);
        assert!(f.message.contains("computeCommandEncoder"), "{}", f.message);
    }
}
