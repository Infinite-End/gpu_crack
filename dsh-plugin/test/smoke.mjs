/**
 * Offline smoke test for the dsh-skill-gpu-crack provider module.
 *
 * It does not boot DSH: it imports the real plugin entry, feeds it a minimal fake
 * Cordis context, and asserts the provider contract used by `ctx.skills`:
 *   - `inject` requires the `skills` service;
 *   - `registerProvider(factory)` receives a synchronous factory whose provider has
 *     a unique name;
 *   - `list()` returns exactly one candidate with the SKILL.md description,
 *     kebab-case name, both invocation surfaces, directory resourceBase and rank;
 *   - `get(candidate, { signal })` returns the markdown body with frontmatter
 *     stripped and the exact `resourceBase` directory;
 *   - relative resources named by the skill exist on disk (`scripts/ctf.py`,
 *     `references/kernels.md`), and the toolkit root is discoverable from the
 *     skill directory (same upward scan `scripts/ctf.py` performs).
 *
 * Run: node test/smoke.mjs
 */
import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const entry = pathToFileURL(join(here, '..', 'lib', 'index.js')).href
const plugin = await import(entry)

const checks = []
function check(name, fn) {
  try {
    fn()
    checks.push(`PASS ${name}`)
  } catch (err) {
    checks.push(`FAIL ${name}: ${err.message}`)
    process.exitCode = 1
  }
}

check('exports cordis plugin shape', () => {
  assert.equal(plugin.name, 'skill-gpu-crack')
  assert.deepEqual(plugin.inject, ['skills'])
  assert.equal(typeof plugin.apply, 'function')
})

let factory
const disposer = () => {}
const ctx = {
  skills: {
    registerProvider(create) {
      factory = create
      return disposer
    },
  },
}
const registered = plugin.apply(ctx, { assetRoot: join(here, '..', '..', 'skill') })

check('registers one provider and returns its disposer', () => {
  assert.equal(registered, disposer)
  assert.equal(typeof factory, 'function')
})

const provider = factory({ signal: new AbortController().signal, invalidate() {} })

check('provider identity', () => {
  assert.equal(provider.name, 'gpu-crack')
})

const candidates = await provider.list({})

check('list() returns the ctf-toolkit candidate', () => {
  assert.equal(candidates.length, 1)
  const [c] = candidates
  assert.equal(c.name, 'ctf-toolkit')
  assert.equal(c.provider, 'gpu-crack')
  assert.equal(c.source, 'bundled')
  assert.ok(c.description.length > 40, 'description should be the SKILL.md one-liner')
  assert.deepEqual(c.invocation, { modelInvocable: true, userInvocable: true })
  assert.equal(c.rank, 600)
  assert.equal(c.resourceBase.kind, 'directory')
  assert.ok(c.path.endsWith('SKILL.md'))
})

const loaded = await provider.get(candidates[0], {})

check('get() returns body without frontmatter', () => {
  assert.equal(loaded.name, 'ctf-toolkit')
  assert.deepEqual(loaded.resourceBase, candidates[0].resourceBase)
  assert.ok(!loaded.content.startsWith('---'), 'frontmatter must be stripped')
  assert.ok(loaded.content.includes('CTF Toolkit'), 'body should contain the skill title')
  assert.equal(loaded.provider, 'gpu-crack')
})

const base = candidates[0].resourceBase.path

check('skill resources exist relative to resourceBase', () => {
  for (const rel of ['SKILL.md', 'SKILL.en.md', 'references/kernels.md', 'scripts/ctf.py']) {
    assert.ok(existsSync(join(base, rel)), `missing ${rel} under ${base}`)
  }
})

check('toolkit root is discoverable upward from the skill dir', () => {
  let dir = base
  let found = false
  for (let i = 0; i < 8; i += 1) {
    if (existsSync(join(dir, 'gpu_crack.py'))) {
      found = true
      break
    }
    const parent = dirname(dir)
    if (parent === dir) break
    dir = parent
  }
  assert.ok(found, 'gpu_crack.py should be found by the dispatcher upward scan')
})

console.log(checks.join('\n'))
console.log(`\nDSH-SKILL-GPU-CRACK: ${checks.every((c) => c.startsWith('PASS')) ? 'ALL PASS' : 'FAIL'}`)
