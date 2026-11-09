/**
 * dsh-skill-gpu-crack — DeepSeek Harness skill provider for the gpu_crack toolkit.
 *
 * Registers one `ctf-toolkit` skill on `ctx.skills` whose body and resources live in
 * this repository's `skill/` directory. The provider is intentionally thin:
 *
 *  - no copy of the skill: `assetRoot` points at the live `skill/` directory, so edits
 *    to `skill/SKILL.md`, `skill/references/kernels.md` or `skill/scripts/ctf.py`
 *    are reflected the next time the skill is loaded;
 *  - `resourceBase` is a directory, so the loaded skill body can resolve relative
 *    resources (`scripts/ctf.py`, `references/kernels.md`) against it — the same
 *    contract the in-box Office skills use;
 *  - frontmatter is stripped from the delivered body and its `description` is used
 *    for the discovery catalog (see SKILL.md frontmatter);
 *  - no `@deepseek-ai/*` package is imported, so the plugin resolves in a plain
 *    profile node_modules without extra peer wiring.
 *
 * Configuration (via this bundle's cordis.patch.yml row, or env fallback):
 *   assetRoot: absolute path to the skill directory (defaults to
 *              `$GPU_CRACK_SKILL_DIR` or `<repo>/skill` relative to this module).
 *
 * @module dsh-skill-gpu-crack
 */
import { readFileSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { dirname, isAbsolute, join } from 'node:path'
import { fileURLToPath } from 'node:url'

/** Skill name shown in the model-facing catalog (kebab-case). */
const SKILL_NAME = 'ctf-toolkit'

/** Provider name; unique within the host layer. */
const PROVIDER_NAME = 'gpu-crack'

/** Precedence rank used by DSH duplicate resolution (packaged/bundled skill). */
const RANK = 600

/**
 * Parse `---` YAML frontmatter without a YAML dependency.
 * Only the flat `key: value` form used by SKILL.md is required here.
 *
 * @param {string} raw SKILL.md source text.
 * @returns {{ metadata: Record<string, string>, body: string }}
 */
function parseFrontmatter(raw) {
  const match = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/u.exec(raw)
  if (match === null) return { metadata: Object.create(null), body: raw.trim() }
  const metadata = Object.create(null)
  for (const line of match[1].split(/\r?\n/u)) {
    const kv = /^([A-Za-z][\w-]*):\s*(.*)$/u.exec(line)
    if (kv !== null) metadata[kv[1]] = kv[2].trim()
  }
  return { metadata, body: raw.slice(match[0].length).trim() }
}

/**
 * Resolve the skill directory: config first, then env, then the repository layout
 * (`<repo>/dsh-plugin/lib/index.js` -> `<repo>/skill`).
 *
 * @param {{ assetRoot?: unknown }} config Bundle row config.
 * @returns {string} Absolute-or-repo-relative skill directory.
 */
function resolveAssetRoot(config) {
  for (const value of [config.assetRoot, process.env.GPU_CRACK_SKILL_DIR]) {
    if (typeof value === 'string' && value.trim() !== '') return value.trim()
  }
  return join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'skill')
}

/** Cordis plugin name. */
const name = 'skill-gpu-crack'

/** Service required by the bundled provider. */
const inject = ['skills']

/**
 * Register the `ctf-toolkit` skill provider on `ctx.skills`.
 *
 * @param {object} ctx Cordis context carrying the skill registry (`ctx.skills`).
 * @param {{ assetRoot?: string }} [config] Bundle row config.
 * @returns {() => void} Registry disposer for the registered provider.
 */
function apply(ctx, config = {}) {
  const assetRoot = resolveAssetRoot(config)
  if (!isAbsolute(assetRoot)) {
    throw new Error(`skill-gpu-crack: assetRoot must be an absolute path, got ${JSON.stringify(assetRoot)}`)
  }
  const skillPath = join(assetRoot, 'SKILL.md')
  const raw = readFileSync(skillPath, 'utf8')
  const { metadata } = parseFrontmatter(raw)
  const description = metadata.description
  if (typeof description !== 'string' || description.length === 0) {
    throw new Error(`skill-gpu-crack: ${skillPath} has no frontmatter description`)
  }

  const summary = {
    name: SKILL_NAME,
    description,
    invocation: { modelInvocable: true, userInvocable: true },
    provider: PROVIDER_NAME,
    source: 'bundled',
    resourceBase: { kind: 'directory', path: assetRoot },
  }
  const candidate = { ...summary, rank: RANK, locator: assetRoot, path: skillPath }

  const provider = {
    name: PROVIDER_NAME,
    list: () => Promise.resolve([candidate]),
    /**
     * Load the skill body. Re-read on every call so repository changes are visible
     * without restarting DSH.
     *
     * @param {{ locator: string }} selected Winning candidate returned by `list()`.
     * @param {{ signal?: AbortSignal } | undefined} options Lookup options.
     */
    async get(selected, options) {
      const body = await readFile(join(selected.locator, 'SKILL.md'), {
        encoding: 'utf8',
        signal: options?.signal,
      })
      return { ...summary, content: parseFrontmatter(body).body }
    },
  }

  return ctx.skills.registerProvider(() => provider)
}

export { apply, inject, name }
