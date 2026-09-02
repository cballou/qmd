/**
 * section-chunking.ts - Section provenance and embedding-noise exclusion.
 *
 * Upstream 2.1.0 replaced our bespoke `##`-splitting chunker with an AST-aware
 * one (tree-sitter for code, heading-scored boundaries for prose), so the
 * splitting half of this module is gone. What upstream still does not carry is
 * *provenance*: its chunks are `{ text, pos, tokens }` with no record of which
 * section a chunk came from, and we embed as `title > section | text` so a
 * heading's words reach the vector even when the chunk body omits them.
 *
 * So this module now maps a chunk's character offset back to its enclosing
 * markdown heading, and keeps the two exclusion policies (noise sections, and
 * aggregate/changelog files) that upstream has no equivalent for.
 */

// =============================================================================
// Section provenance
// =============================================================================

/** A markdown heading and the character offset at which its body starts. */
export interface SectionMarker {
  /** Character offset of the heading line in the source document. */
  pos: number;
  /** Heading text, e.g. "Core Rules". */
  heading: string;
}

/** Matches ATX headings at level 2+ (`## ...`), which is how our notes divide. */
const SECTION_HEADING = /^(#{2,6})\s+(.+?)\s*#*\s*$/gm;

/** Fenced code blocks — `## ` inside one is a comment, not a heading. */
const FENCE = /^(?:```|~~~)/gm;

/**
 * Character ranges covered by fenced code blocks, so heading detection can
 * skip them. An unterminated fence runs to end of document.
 */
function fencedRanges(content: string): [number, number][] {
  const ranges: [number, number][] = [];
  FENCE.lastIndex = 0;
  let open: number | null = null;
  for (let m = FENCE.exec(content); m; m = FENCE.exec(content)) {
    if (open === null) open = m.index;
    else {
      ranges.push([open, m.index + m[0].length]);
      open = null;
    }
  }
  if (open !== null) ranges.push([open, content.length]);
  return ranges;
}

/**
 * Build the ordered list of section markers for a document.
 *
 * Built once per document and reused across its chunks: doing this per chunk
 * would rescan the whole document for every chunk it produced.
 */
export function buildSectionMap(content: string): SectionMarker[] {
  const fences = fencedRanges(content);
  const inFence = (pos: number) => fences.some(([a, b]) => pos >= a && pos < b);

  const markers: SectionMarker[] = [];
  SECTION_HEADING.lastIndex = 0;
  for (let m = SECTION_HEADING.exec(content); m; m = SECTION_HEADING.exec(content)) {
    if (inFence(m.index)) continue;
    markers.push({ pos: m.index, heading: m[2]!.trim() });
  }
  return markers;
}

/**
 * The section a chunk starting at `pos` belongs to, or "" for preamble content
 * ahead of the first heading.
 *
 * Upstream's chunker prefers heading boundaries but does not guarantee them, so
 * a chunk can straddle two sections; we attribute it to the section it starts
 * in. Binary search keeps this O(log n) per chunk.
 */
export function sectionAtPosition(markers: SectionMarker[], pos: number): string {
  let lo = 0;
  let hi = markers.length - 1;
  let found = "";
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (markers[mid]!.pos <= pos) {
      found = markers[mid]!.heading;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

// =============================================================================
// Embedding-noise exclusion
// =============================================================================

/**
 * Sections that are bookkeeping rather than content. They are still reachable
 * by keyword search via FTS; excluding them keeps vector search from surfacing
 * a note's changelog instead of its substance.
 */
const EXCLUDED_SECTION_PATTERNS = [
  /^recent\s+activity$/i,
  /^evolution\s+timeline$/i,
  /^related$/i,
  /^changelog$/i,
  /^version\s+history$/i,
];

/** Whether a section heading names bookkeeping we do not want embedded. */
export function shouldExcludeSection(heading: string): boolean {
  return EXCLUDED_SECTION_PATTERNS.some(pat => pat.test(heading.trim()));
}

/**
 * Patterns for files that should be excluded from vector embedding.
 * These files add noise to semantic search (changelogs, aggregations, etc.)
 * but are still valuable for FTS keyword search.
 */
const EXCLUDED_FILE_PATTERNS = [
  /changelog/i,
  /CHANGELOG/,
  /-changes\./,
  /-log\./,
  /aggregate/i,
  /combined-notes/i,
];

/**
 * Check if a file path should be excluded from vector embedding.
 * Returns true for changelog/aggregate files that add noise to semantic search.
 */
export function shouldExcludeFile(path: string): boolean {
  const basename = path.split("/").pop() || path;
  return EXCLUDED_FILE_PATTERNS.some(pat => pat.test(basename));
}
