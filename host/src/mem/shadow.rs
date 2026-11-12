//! The shadow/diff protocol for memory that the CPU and the GPU both see (plan section 6.2).
//!
//! In Metal a shared-storage buffer is one copy of memory. Here the guest's pages are authoritative for the CPU and a
//! WebGPU buffer for the GPU, and a [`Shadow`] holds the bytes the two last agreed on:
//!
//! - **Upload.** Before the GPU uses the memory, every byte where the guest differs from the shadow is written to the
//!   GPU copy ([`Shadow::plan_upload`], [`Shadow::apply_upload`]), and the shadow takes the guest's value.
//! - **Readback.** After GPU work that may have written the memory, the GPU copy is read back and every byte where it
//!   differs from what the GPU held before that work is written to the guest ([`Shadow::merge_readback`]), and the
//!   shadow takes it.
//!
//! Diffing in both directions is what keeps CPU writes to other bytes of the same memory, made while the GPU works,
//! intact: Metal allows those (it forbids only CPU and GPU access to the same bytes at the same time), and only the
//! bytes each side changed travel. Programs that race the CPU and the GPU on the same bytes get one of the results
//! their race allows (plan section 9, row 6).
//!
//! **GPU-timeline order.** Uploads and readbacks are positioned on the WebGPU queue by a sequence number: an upload
//! with a higher number than a readback reaches the GPU after that readback's copy. A readback's data arrive
//! (`mapAsync`) after later uploads may already have been applied to the shadow, so the shadow logs every upload made
//! while a readback is pending, with the bytes it replaced; [`Shadow::merge_readback`] compares the readback with the
//! shadow as it was at the readback's position. Bytes a later upload covers are not merged: the CPU wrote them after
//! the GPU work was committed, which is the race Metal forbids, and the CPU's value is the one kept.
//!
//! **Alignment.** WebGPU writes buffers in 4-byte units (`writeBuffer` offset and size, WebGPU section 21.2.1, and the
//! buffer-copy rules of `tables/limits.toml` question `6.3-copy-alignment`). A changed run is widened to whole words;
//! the bytes added are ones the guest did not change, so writing the guest's value of them is exact as long as no GPU
//! work that could change them is still pending. When some is, [`Shadow::plan_upload`] reports
//! [`UploadPlan::Blocked`] instead: the caller first completes those readbacks, after which the widened bytes hold
//! their final values in the shadow and the guest alike.

use std::collections::BTreeSet;

/// Bytes `[start, end)` of the memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub start: u64,
    pub end: u64,
}

/// One write of the guest's bytes to the GPU copy: 4-byte aligned offset and length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UploadWrite {
    pub offset: u64,
    pub bytes: Vec<u8>,
}

/// What [`Shadow::plan_upload`] decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UploadPlan {
    /// Write these runs (possibly none).
    Writes(Vec<UploadWrite>),
    /// A run had to be widened over bytes that pending GPU work may write; complete the readbacks with these
    /// sequence numbers, then plan again.
    Blocked(Vec<u64>),
}

/// A run to write into the guest after a readback: `new` replaces the bytes of the guest where it differs from
/// `old`, the GPU copy's bytes before the work that was read back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuestWrite {
    pub offset: u64,
    pub new: Vec<u8>,
    pub old: Vec<u8>,
}

#[derive(Clone, Debug)]
struct Logged {
    seq: u64,
    offset: usize,
    replaced: Vec<u8>,
}

/// The shadow of `len` bytes of CPU-visible memory.
#[derive(Clone, Debug)]
pub struct Shadow {
    len: u64,
    /// `len` rounded up to whole words; the bytes past `len` are always zero.
    bytes: Vec<u8>,
    /// Uploads applied while a readback was pending, oldest first.
    log: Vec<Logged>,
    /// Readbacks submitted and not merged yet.
    pending: BTreeSet<u64>,
}

impl Shadow {
    /// The shadow of memory whose GPU copy and guest bytes are both `initial` (zero for a new Metal buffer: Metal and
    /// WebGPU both zero-fill new buffers, `tables/limits.toml` question `6.10-new-buffer-contents`).
    pub fn new(len: u64, initial: Option<&[u8]>) -> Shadow {
        let mut bytes = vec![0u8; len.next_multiple_of(4) as usize];
        if let Some(init) = initial {
            assert_eq!(init.len() as u64, len, "initial bytes must cover the memory");
            bytes[..init.len()].copy_from_slice(init);
        }
        Shadow { len, bytes, log: Vec::new(), pending: BTreeSet::new() }
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The length of the GPU copy: `len` rounded up to whole words.
    pub fn padded_len(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// The bytes the guest and the GPU last agreed on.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }

    /// Readbacks submitted and not merged.
    pub fn pending(&self) -> impl Iterator<Item = u64> + '_ {
        self.pending.iter().copied()
    }

    /// The runs where `chunks` (each the guest's bytes at an offset of the memory) differ from the shadow.
    pub fn changed_runs(&self, chunks: &[(u64, &[u8])]) -> Vec<Run> {
        let mut runs = Vec::new();
        for &(offset, guest) in chunks {
            assert!(offset + guest.len() as u64 <= self.len, "guest chunk outside the memory");
            let shadow = &self.bytes[offset as usize..offset as usize + guest.len()];
            let mut at = 0;
            while at < guest.len() {
                if guest[at] == shadow[at] {
                    at += 1;
                    continue;
                }
                let start = at;
                while at < guest.len() && guest[at] != shadow[at] {
                    at += 1;
                }
                runs.push(Run { start: offset + start as u64, end: offset + at as u64 });
            }
        }
        runs.sort_by_key(|r| r.start);
        runs
    }

    /// Plan the upload of the guest's changes. `chunks` are the guest's bytes at offsets of the memory; together they
    /// cover every byte the guest may have changed since the previous upload (all of the memory, or the pages the
    /// embedder reports written).
    pub fn plan_upload(&self, chunks: &[(u64, &[u8])]) -> UploadPlan {
        let runs = self.changed_runs(chunks);
        if runs.is_empty() {
            return UploadPlan::Writes(Vec::new());
        }
        // Widen to whole words and join runs that then touch or overlap.
        let mut words: Vec<Run> = Vec::new();
        for r in &runs {
            let w = Run { start: r.start & !3, end: r.end.next_multiple_of(4) };
            match words.last_mut() {
                Some(last) if w.start <= last.end => last.end = last.end.max(w.end),
                _ => words.push(w),
            }
        }
        let widened: u64 = words.iter().map(|w| w.end - w.start).sum::<u64>();
        let changed: u64 = runs.iter().map(|r| r.end - r.start).sum::<u64>();
        if widened != changed && !self.pending.is_empty() {
            return UploadPlan::Blocked(self.pending.iter().copied().collect());
        }
        // The guest's value of each byte of a word: changed bytes come from the chunks, the bytes widening added are
        // unchanged, so the shadow holds the guest's value of them.
        let mut writes = Vec::with_capacity(words.len());
        for w in words {
            let mut bytes = self.bytes[w.start as usize..w.end as usize].to_vec();
            for &(offset, guest) in chunks {
                let lo = offset.max(w.start);
                let hi = (offset + guest.len() as u64).min(w.end);
                if lo < hi {
                    bytes[(lo - w.start) as usize..(hi - w.start) as usize]
                        .copy_from_slice(&guest[(lo - offset) as usize..(hi - offset) as usize]);
                }
            }
            writes.push(UploadWrite { offset: w.start, bytes });
        }
        UploadPlan::Writes(writes)
    }

    /// Record that `writes` were issued to the GPU at queue position `seq`, after every readback still pending.
    pub fn apply_upload(&mut self, writes: &[UploadWrite], seq: u64) {
        let logging = !self.pending.is_empty();
        debug_assert!(self.pending.iter().all(|&p| p < seq), "an upload must follow the pending readbacks");
        for w in writes {
            let (start, end) = (w.offset as usize, w.offset as usize + w.bytes.len());
            if logging {
                self.log.push(Logged { seq, offset: start, replaced: self.bytes[start..end].to_vec() });
            }
            self.bytes[start..end].copy_from_slice(&w.bytes);
        }
    }

    /// Record that the GPU copy is read back at queue position `seq`.
    pub fn begin_readback(&mut self, seq: u64) {
        debug_assert!(self.log.iter().all(|l| l.seq < seq), "a readback must follow the uploads already applied");
        self.pending.insert(seq);
    }

    /// Merge the GPU copy as read back at `seq` (`data`, the whole padded copy). Returns the runs to write into the
    /// guest; the shadow already holds them.
    pub fn merge_readback(&mut self, seq: u64, data: &[u8]) -> Vec<GuestWrite> {
        assert!(self.pending.remove(&seq), "readback {seq} is not pending");
        assert_eq!(data.len(), self.bytes.len(), "a readback covers the whole GPU copy");
        // The GPU copy before the work read back: the shadow without the uploads that reached the GPU later.
        let mut base = self.bytes.clone();
        let mut later = vec![false; self.bytes.len()];
        for l in self.log.iter().rev().filter(|l| l.seq > seq) {
            base[l.offset..l.offset + l.replaced.len()].copy_from_slice(&l.replaced);
            later[l.offset..l.offset + l.replaced.len()].fill(true);
        }
        let mut writes = Vec::new();
        let len = self.len as usize;
        let mut at = 0;
        while at < len {
            if data[at] == base[at] || later[at] {
                at += 1;
                continue;
            }
            let start = at;
            while at < len && data[at] != base[at] && !later[at] {
                at += 1;
            }
            self.bytes[start..at].copy_from_slice(&data[start..at]);
            writes.push(GuestWrite {
                offset: start as u64,
                new: data[start..at].to_vec(),
                old: base[start..at].to_vec(),
            });
        }
        // Keep the log entries a pending readback still needs.
        match self.pending.first() {
            Some(&oldest) => self.log.retain(|l| l.seq > oldest),
            None => self.log.clear(),
        }
        writes
    }

    /// The contents were discarded and replaced by zeros on both sides (`setPurgeableState:` empty).
    pub fn reset_zero(&mut self) {
        assert!(self.pending.is_empty(), "contents discarded while a readback is pending");
        self.bytes.fill(0);
        self.log.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_widens_to_words() {
        let s = Shadow::new(10, None);
        let guest = [0, 0, 0, 0, 0, 7, 0, 0, 0, 9];
        let UploadPlan::Writes(w) = s.plan_upload(&[(0, &guest)]) else { panic!() };
        // Words 4..8 and 8..12 touch, so they go as one write.
        assert_eq!(w, vec![UploadWrite { offset: 4, bytes: vec![0, 7, 0, 0, 0, 9, 0, 0] }]);
        let guest = [0, 0, 1, 0, 0, 0, 0, 0, 0, 9];
        let UploadPlan::Writes(w) = s.plan_upload(&[(0, &guest)]) else { panic!() };
        assert_eq!(
            w,
            vec![
                UploadWrite { offset: 0, bytes: vec![0, 0, 1, 0] },
                UploadWrite { offset: 8, bytes: vec![0, 9, 0, 0] }
            ]
        );
    }

    #[test]
    fn upload_from_dirty_chunks() {
        let s = Shadow::new(16, Some(&[1; 16]));
        // Only bytes 8..12 were reported written; byte 9 changed.
        let UploadPlan::Writes(w) = s.plan_upload(&[(8, &[1, 5, 1, 1])]) else { panic!() };
        assert_eq!(w, vec![UploadWrite { offset: 8, bytes: vec![1, 5, 1, 1] }]);
    }

    #[test]
    fn widening_waits_for_pending_writers() {
        let mut s = Shadow::new(8, None);
        s.begin_readback(1);
        // An aligned change can go; an unaligned one must wait for readback 1.
        assert!(matches!(s.plan_upload(&[(0, &[1, 2, 3, 4, 0, 0, 0, 0])]), UploadPlan::Writes(_)));
        assert_eq!(s.plan_upload(&[(0, &[0, 0, 0, 0, 0, 6, 0, 0])]), UploadPlan::Blocked(vec![1]));
    }

    #[test]
    fn readback_compares_with_the_gpu_timeline() {
        let mut s = Shadow::new(8, None);
        // Command buffer 1 writes bytes 0..4 and is read back at 2; before its data arrive the CPU changes bytes 4..8
        // and they are uploaded at 3.
        s.begin_readback(2);
        let UploadPlan::Writes(w) = s.plan_upload(&[(0, &[0, 0, 0, 0, 9, 9, 9, 9])]) else { panic!() };
        s.apply_upload(&w, 3);
        // The readback holds the GPU work's bytes and the old bytes 4..8.
        let g = s.merge_readback(2, &[5, 5, 5, 5, 0, 0, 0, 0]);
        assert_eq!(g, vec![GuestWrite { offset: 0, new: vec![5; 4], old: vec![0; 4] }]);
        assert_eq!(s.bytes(), &[5, 5, 5, 5, 9, 9, 9, 9]);
        assert!(s.log.is_empty());
    }

    #[test]
    fn racing_bytes_keep_the_cpu_value() {
        let mut s = Shadow::new(4, None);
        s.begin_readback(1);
        let UploadPlan::Writes(w) = s.plan_upload(&[(0, &[3, 3, 3, 3])]) else { panic!() };
        s.apply_upload(&w, 2);
        // The GPU work of 1 also wrote these bytes: the race is resolved in favour of the later CPU write.
        assert!(s.merge_readback(1, &[7, 7, 7, 7]).is_empty());
        assert_eq!(s.bytes(), &[3, 3, 3, 3]);
    }

    /// A random legal program run against the shared-memory model.
    #[test]
    fn property_random_programs_match_shared_memory() {
        let mut stats = sim::Stats::default();
        for seed in 1..=400u64 {
            sim::run(seed, 64 + (seed % 61) as usize, 120, &mut stats);
        }
        eprintln!("shadow property test: {stats:?}");
        // The programs exercise the paths that make the protocol exact, not only the easy ones.
        assert!(stats.blocked > 100, "{stats:?}");
        assert!(stats.merges_after_later_uploads > 100, "{stats:?}");
        assert!(stats.unaligned_writes > 1000, "{stats:?}");
    }

    /// A simulation of the executor's use of the protocol over a GPU that runs queued work in order and late.
    mod sim {
        use super::super::*;

        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0
            }
            fn below(&mut self, n: u64) -> u64 {
                self.next() % n
            }
        }

        enum GpuOp {
            Write(UploadWrite),
            /// A command buffer's work: add `delta` to each byte of the range (reads and writes it).
            Add {
                start: usize,
                end: usize,
                delta: u8,
            },
            Fill {
                start: usize,
                end: usize,
                value: u8,
            },
            Readback(u64),
        }

        #[derive(Debug, Default)]
        pub struct Stats {
            pub blocked: u64,
            pub merges_after_later_uploads: u64,
            pub unaligned_writes: u64,
        }

        struct Committed {
            seq: u64,
            start: usize,
            end: usize,
        }

        pub fn run(seed: u64, len: usize, steps: usize, stats: &mut Stats) {
            let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
            let mut model = vec![0u8; len];
            let mut guest = vec![0u8; len];
            let mut gpu = vec![0u8; len.next_multiple_of(4)];
            let mut shadow = Shadow::new(len as u64, None);
            let mut queue: Vec<GpuOp> = Vec::new();
            let mut readbacks: Vec<(u64, Vec<u8>)> = Vec::new();
            let mut in_flight: Vec<Committed> = Vec::new();
            let mut seq = 0u64;

            let run_gpu = |queue: &mut Vec<GpuOp>, gpu: &mut Vec<u8>, readbacks: &mut Vec<(u64, Vec<u8>)>, n: usize| {
                for op in queue.drain(..n.min(queue.len())) {
                    match op {
                        GpuOp::Write(w) => {
                            gpu[w.offset as usize..w.offset as usize + w.bytes.len()].copy_from_slice(&w.bytes)
                        }
                        GpuOp::Add { start, end, delta } => {
                            for b in &mut gpu[start..end] {
                                *b = b.wrapping_add(delta);
                            }
                        }
                        GpuOp::Fill { start, end, value } => gpu[start..end].fill(value),
                        GpuOp::Readback(s) => readbacks.push((s, gpu.clone())),
                    }
                }
            };
            // Complete the oldest in-flight command buffer: run the GPU until its readback is done, then merge.
            let complete_oldest = |queue: &mut Vec<GpuOp>,
                                   gpu: &mut Vec<u8>,
                                   readbacks: &mut Vec<(u64, Vec<u8>)>,
                                   in_flight: &mut Vec<Committed>,
                                   shadow: &mut Shadow,
                                   guest: &mut Vec<u8>| {
                let c = in_flight.remove(0);
                while !readbacks.iter().any(|(s, _)| *s == c.seq) {
                    run_gpu(queue, gpu, readbacks, 1);
                }
                let i = readbacks.iter().position(|(s, _)| *s == c.seq).unwrap();
                let (_, data) = readbacks.remove(i);
                let later = shadow.log.iter().any(|l| l.seq > c.seq);
                for w in shadow.merge_readback(c.seq, &data) {
                    for (k, (&n, &o)) in w.new.iter().zip(&w.old).enumerate() {
                        if n != o {
                            guest[w.offset as usize + k] = n;
                        }
                    }
                }
                later
            };

            for _ in 0..steps {
                match rng.below(5) {
                    // CPU writes bytes no in-flight command buffer touches, at any alignment.
                    0 | 1 => {
                        let start = rng.below(len as u64) as usize;
                        let end = (start + 1 + rng.below(9) as usize).min(len);
                        if in_flight.iter().any(|c| start < c.end && c.start < end) {
                            continue;
                        }
                        if !start.is_multiple_of(4) || !end.is_multiple_of(4) {
                            stats.unaligned_writes += 1;
                        }
                        for i in start..end {
                            let v = rng.next() as u8;
                            guest[i] = v;
                            model[i] = v;
                        }
                    }
                    // Commit a command buffer whose GPU work reads and writes a range nobody else is using.
                    2 | 3 => {
                        let start = rng.below(len as u64) as usize;
                        let end = (start + 1 + rng.below(12) as usize).min(len);
                        if in_flight.iter().any(|c| start < c.end && c.start < end) {
                            continue;
                        }
                        // Upload, completing pending readbacks first when the plan asks for it.
                        let writes = loop {
                            match shadow.plan_upload(&[(0, &guest)]) {
                                UploadPlan::Writes(w) => break w,
                                UploadPlan::Blocked(_) => {
                                    stats.blocked += 1;
                                    stats.merges_after_later_uploads += complete_oldest(
                                        &mut queue,
                                        &mut gpu,
                                        &mut readbacks,
                                        &mut in_flight,
                                        &mut shadow,
                                        &mut guest,
                                    ) as u64;
                                }
                            }
                        };
                        seq += 1;
                        shadow.apply_upload(&writes, seq);
                        queue.extend(writes.into_iter().map(GpuOp::Write));
                        if rng.below(2) == 0 {
                            let delta = 1 + rng.below(250) as u8;
                            queue.push(GpuOp::Add { start, end, delta });
                            for b in &mut model[start..end] {
                                *b = b.wrapping_add(delta);
                            }
                        } else {
                            let value = rng.next() as u8;
                            queue.push(GpuOp::Fill { start, end, value });
                            model[start..end].fill(value);
                        }
                        seq += 1;
                        queue.push(GpuOp::Readback(seq));
                        shadow.begin_readback(seq);
                        in_flight.push(Committed { seq, start, end });
                    }
                    // The GPU makes progress, or the oldest command buffer completes.
                    _ => {
                        if rng.below(2) == 0 {
                            let n = rng.below(4) as usize;
                            run_gpu(&mut queue, &mut gpu, &mut readbacks, n);
                        } else if !in_flight.is_empty() {
                            let c_start = in_flight[0].start;
                            let c_end = in_flight[0].end;
                            stats.merges_after_later_uploads += complete_oldest(
                                &mut queue,
                                &mut gpu,
                                &mut readbacks,
                                &mut in_flight,
                                &mut shadow,
                                &mut guest,
                            ) as u64;
                            assert_eq!(guest[c_start..c_end], model[c_start..c_end], "seed {seed}: completed range");
                        }
                    }
                }
            }
            while !in_flight.is_empty() {
                stats.merges_after_later_uploads +=
                    complete_oldest(&mut queue, &mut gpu, &mut readbacks, &mut in_flight, &mut shadow, &mut guest)
                        as u64;
            }
            assert_eq!(guest, model, "seed {seed}: guest memory after every command buffer completed");
            // The next use uploads the CPU's last writes; the GPU copy then equals the model too.
            let UploadPlan::Writes(writes) = shadow.plan_upload(&[(0, &guest)]) else { panic!("nothing is pending") };
            seq += 1;
            shadow.apply_upload(&writes, seq);
            queue.extend(writes.into_iter().map(GpuOp::Write));
            run_gpu(&mut queue, &mut gpu, &mut readbacks, usize::MAX);
            assert_eq!(shadow.bytes(), &model[..], "seed {seed}: shadow");
            assert_eq!(gpu[..len], model[..], "seed {seed}: GPU copy");
        }
    }
}
