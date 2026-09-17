import * as fs from 'node:fs';
import * as path from 'node:path';

import type { RagTestCase, RagTestCaseMetadata } from '../models/ragTestCase';

/** Golden dataset produced by `scripts/prepareDataset.py`. */
export const DEFAULT_DATASET_PATH = path.resolve(
  __dirname,
  '..',
  '..',
  'data',
  'golden',
  'rag_test_cases.json',
);

/** Raised when the golden file is missing, unparseable, or fails validation. */
export class DatasetValidationError extends Error {
  constructor(
    message: string,
    readonly issues: readonly string[] = [],
  ) {
    const detail = issues.length > 0 ? `\n  - ${issues.join('\n  - ')}` : '';
    super(`${message}${detail}`);
    this.name = 'DatasetValidationError';
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

/** Validates an array of strings, requiring every entry to be non-empty. */
function checkStringArray(
  value: unknown,
  field: string,
  where: string,
  issues: string[],
): value is readonly string[] {
  if (!Array.isArray(value)) {
    issues.push(`${where}: "${field}" must be an array`);
    return false;
  }
  const bad = value.findIndex((entry) => !isNonEmptyString(entry));
  if (bad !== -1) {
    issues.push(`${where}: "${field}[${bad}]" must be a non-empty string`);
    return false;
  }
  return true;
}

function checkMetadata(value: unknown, where: string, issues: string[]): boolean {
  if (!isRecord(value)) {
    issues.push(`${where}: "metadata" must be an object`);
    return false;
  }
  let ok = true;
  for (const field of ['squadId', 'title', 'split'] satisfies (keyof RagTestCaseMetadata)[]) {
    if (!isNonEmptyString(value[field])) {
      issues.push(`${where}: "metadata.${field}" must be a non-empty string`);
      ok = false;
    }
  }
  return ok;
}

/**
 * Validates one entry and returns it typed, or `undefined` if it is malformed.
 *
 * All problems with the entry are appended to `issues` rather than thrown, so a
 * single run reports every fault in the file instead of only the first.
 */
function parseTestCase(value: unknown, index: number, issues: string[]): RagTestCase | undefined {
  const where = `case[${index}]`;
  if (!isRecord(value)) {
    issues.push(`${where}: expected an object`);
    return undefined;
  }

  let ok = true;
  for (const field of ['id', 'question', 'source'] as const) {
    if (!isNonEmptyString(value[field])) {
      issues.push(`${where}: "${field}" must be a non-empty string`);
      ok = false;
    }
  }
  if (typeof value['answerable'] !== 'boolean') {
    issues.push(`${where}: "answerable" must be a boolean`);
    ok = false;
  }

  const contextOk = checkStringArray(value['context'], 'context', where, issues);
  if (contextOk && (value['context'] as readonly string[]).length === 0) {
    issues.push(`${where}: "context" must not be empty`);
    ok = false;
  }
  const groundTruthOk = checkStringArray(value['groundTruth'], 'groundTruth', where, issues);
  ok = ok && contextOk && groundTruthOk && checkMetadata(value['metadata'], where, issues);
  if (!ok) {
    return undefined;
  }

  const testCase = value as unknown as RagTestCase;

  // The answerable flag and groundTruth must agree, otherwise scoring in later
  // milestones would silently treat an abstention as a miss (or vice versa).
  if (testCase.answerable && testCase.groundTruth.length === 0) {
    issues.push(`${where} (${testCase.id}): answerable case has no groundTruth`);
    return undefined;
  }
  if (!testCase.answerable && testCase.groundTruth.length > 0) {
    issues.push(`${where} (${testCase.id}): unanswerable case must have an empty groundTruth`);
    return undefined;
  }

  return testCase;
}

/**
 * Reads, parses and validates the golden dataset.
 *
 * @throws {@link DatasetValidationError} if the file is missing, is not valid
 * JSON, is not an array, contains a malformed case, or repeats an id.
 */
export function loadRagTestCases(datasetPath: string = DEFAULT_DATASET_PATH): RagTestCase[] {
  let raw: string;
  try {
    raw = fs.readFileSync(datasetPath, 'utf-8');
  } catch (cause) {
    throw new DatasetValidationError(
      `Unable to read golden dataset at ${datasetPath}. ` +
        'Run `python scripts/prepareDataset.py` to generate it.',
      [cause instanceof Error ? cause.message : String(cause)],
    );
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (cause) {
    throw new DatasetValidationError(`Golden dataset at ${datasetPath} is not valid JSON.`, [
      cause instanceof Error ? cause.message : String(cause),
    ]);
  }

  if (!Array.isArray(parsed)) {
    throw new DatasetValidationError(
      `Golden dataset at ${datasetPath} must contain a JSON array of test cases.`,
    );
  }

  const issues: string[] = [];
  const testCases: RagTestCase[] = [];
  const seenIds = new Set<string>();

  parsed.forEach((entry, index) => {
    const testCase = parseTestCase(entry, index, issues);
    if (!testCase) {
      return;
    }
    if (seenIds.has(testCase.id)) {
      issues.push(`case[${index}]: duplicate id "${testCase.id}"`);
      return;
    }
    seenIds.add(testCase.id);
    testCases.push(testCase);
  });

  if (issues.length > 0) {
    throw new DatasetValidationError(
      `Golden dataset at ${datasetPath} failed validation (${issues.length} issue(s)).`,
      issues,
    );
  }

  return testCases;
}