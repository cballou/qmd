/**
 * section-chunking.ts - Section-aware document chunking for markdown files.
 *
 * Splits markdown documents by ## headings into section-aware chunks,
 * preserving section context for better embedding quality.
 * Also provides file exclusion logic for aggregate/changelog files
 * that add noise to vector search.
 */

import { getDefaultLlamaCpp } from "./llm.js";
import { CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS } from "./store.js";

// =============================================================================
// Section-Aware Chunking
// =============================================================================

export interface SectionChunk {
  section: string;   // Section heading (e.g. "Core Rules"), empty for preamble
  text: string;      // Chunk text content
  pos: number;       // Character position in original document
  tokens: number;    // Estimated token count
  bytes: number;     // Byte length of text
}

/**
 * Estimate token count from text.
 * Uses a conservative ~4 chars per token for prose, which is a reasonable
 * approximation for markdown content without requiring the LLM tokenizer.
 */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Count actual tokens using the LLM tokenizer if available,
 * falling back to estimation.
 */
async function countTokens(text: string): Promise<number> {
  try {
    const llm = getDefaultLlamaCpp();
    const count = await llm.countTokens(text);
    return count ?? estimateTokens(text);
  } catch {
    return estimateTokens(text);
  }
}

/**
 * Sections to exclude from embedding because they contain noise
 * (boilerplate, auto-generated content, etc.)
 */
const EXCLUDED_SECTION_PATTERNS = [
  /^recent\s+activity$/i,
  /^evolution\s+timeline$/i,
  /^related$/i,
  /^changelog$/i,
  /^version\s+history$/i,
];

function shouldExcludeSection(heading: string): boolean {
  return EXCLUDED_SECTION_PATTERNS.some(pat => pat.test(heading.trim()));
}

/**
 * Split a markdown document into section-aware chunks.
 *
 * Strategy:
 * 1. Split by ## headings (level 2) into logical sections
 * 2. Each section becomes one chunk if it fits within token budget
 * 3. Oversized sections are split at paragraph boundaries with overlap
 * 4. Preamble (content before first ##) gets section=""
 * 5. Sections matching noise patterns (Recent Activity, etc.) are excluded
 *
 * Returns chunks with section names for embedding context.
 */
export async function chunkDocumentBySections(
  body: string,
  maxTokens: number = CHUNK_SIZE_TOKENS,
  overlapTokens: number = CHUNK_OVERLAP_TOKENS,
): Promise<SectionChunk[]> {
  const encoder = new TextEncoder();
  const chunks: SectionChunk[] = [];

  // Split by ## headings, keeping the heading with its content
  const sectionPattern = /^## (.+)$/gm;
  const sections: { heading: string; text: string; pos: number }[] = [];

  let lastIndex = 0;
  let lastHeading = "";
  let match: RegExpExecArray | null;

  // Collect all ## heading positions
  const headingMatches: { heading: string; index: number }[] = [];
  while ((match = sectionPattern.exec(body)) !== null) {
    headingMatches.push({ heading: match[1]!, index: match.index });
  }

  // Build sections from heading positions
  for (let i = 0; i < headingMatches.length; i++) {
    const hm = headingMatches[i]!;
    // Content before this heading belongs to previous section
    if (hm.index > lastIndex) {
      sections.push({
        heading: lastHeading,
        text: body.slice(lastIndex, hm.index).trim(),
        pos: lastIndex,
      });
    }
    lastHeading = hm.heading;
    lastIndex = hm.index;
  }

  // Remaining content after last heading (or entire doc if no headings)
  if (lastIndex < body.length) {
    sections.push({
      heading: lastHeading,
      text: body.slice(lastIndex).trim(),
      pos: lastIndex,
    });
  }

  // Process each section
  for (const section of sections) {
    if (!section.text) continue;

    // Skip noise sections
    if (section.heading && shouldExcludeSection(section.heading)) continue;

    const tokens = await countTokens(section.text);

    if (tokens <= maxTokens) {
      // Section fits in one chunk
      chunks.push({
        section: section.heading,
        text: section.text,
        pos: section.pos,
        tokens,
        bytes: encoder.encode(section.text).length,
      });
    } else {
      // Section too large — split at paragraph boundaries
      const subChunks = await splitSectionByParagraphs(
        section.text,
        section.heading,
        section.pos,
        maxTokens,
        overlapTokens,
      );
      chunks.push(...subChunks);
    }
  }

  // If no chunks produced (empty doc or all sections excluded), return single chunk
  if (chunks.length === 0 && body.trim()) {
    const text = body.trim();
    const tokens = await countTokens(text);
    chunks.push({
      section: "",
      text,
      pos: 0,
      tokens,
      bytes: encoder.encode(text).length,
    });
  }

  return chunks;
}

/**
 * Split an oversized section at paragraph boundaries (\n\n).
 * Adds overlap between chunks for context continuity.
 */
async function splitSectionByParagraphs(
  text: string,
  heading: string,
  basePos: number,
  maxTokens: number,
  overlapTokens: number,
): Promise<SectionChunk[]> {
  const encoder = new TextEncoder();
  const paragraphs = text.split(/\n\n+/);
  const chunks: SectionChunk[] = [];

  let currentText = "";
  let currentTokens = 0;
  let currentPos = basePos;

  for (const para of paragraphs) {
    const paraTokens = await countTokens(para);

    if (currentTokens + paraTokens > maxTokens && currentText) {
      // Emit current chunk
      const trimmed = currentText.trim();
      chunks.push({
        section: heading,
        text: trimmed,
        pos: currentPos,
        tokens: currentTokens,
        bytes: encoder.encode(trimmed).length,
      });

      // Start new chunk with overlap: keep the last paragraph for context
      const overlapText = currentText.slice(-(overlapTokens * 4));
      currentText = overlapText + "\n\n" + para;
      currentTokens = await countTokens(currentText);
      currentPos = basePos + text.indexOf(para);
    } else {
      currentText += (currentText ? "\n\n" : "") + para;
      currentTokens += paraTokens;
    }
  }

  // Emit final chunk
  if (currentText.trim()) {
    const trimmed = currentText.trim();
    chunks.push({
      section: heading,
      text: trimmed,
      pos: currentPos,
      tokens: await countTokens(trimmed),
      bytes: encoder.encode(trimmed).length,
    });
  }

  return chunks;
}

// =============================================================================
// File Exclusion
// =============================================================================

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
