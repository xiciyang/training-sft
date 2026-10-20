#!/usr/bin/env node
/**
 * docker-worker.spec（**真机**）—— 验证 ContainerWorker 真的能起容器、走协议、可清理。
 * CI 不跑它（tests/run.mjs 不收集本目录）。
 */
import { execFileSync } from 'node:child_process'
import { ContainerWorker } from '../../src/worker.mjs'

let failed = 0
const t = { assert(c, m) { console.log(`${c ? 'PASS' : 'FAIL'} ${m}`); if (!c) failed += 1 } }

const image = 'training-sft/toolchain:0.1.0'
let imgOk = false
try { imgOk = execFileSync('docker', ['images', '-q', image], { encoding: 'utf8' }).trim().length > 0 } catch { imgOk = false }
t.assert(imgOk, `镜像存在：${image}`)
if (!imgOk) { console.log('镜像不存在 → 跳过（这是真机测试，未通过即视为失败）'); process.exit(1) }

const w = new ContainerWorker({ image, mounts: ['<MODELS_DIR>:/models:ro'], gpus: true, entry: '/work/trainer.py', python: 'python3', log: (m) => console.log('  [w]', m) })
try {
  w.start()
  const p = await w.call('ping', {}, { timeoutMs: 120000 })
  t.assert(p.ok === true && p.pong === true, `ping 通过（torch=${p.torch} cuda=${p.cuda}）`)
  t.assert(p.cuda === true, '容器内可见 GPU')
  await w.call('shutdown', {}, { timeoutMs: 20000 }).catch(() => {})
} finally {
  await w.stop()
}
const leftovers = execFileSync('docker', ['ps', '-a', '--filter', `ancestor=${image}`, '--format', '{{.Names}}'], { encoding: 'utf8' }).trim()
t.assert(leftovers === '', `无容器残留（${leftovers || 'none'}）`)
console.log(failed === 0 ? '\nREALMACHINE WORKER: PASSED' : `\nREALMACHINE WORKER: ${failed} FAILED`)
process.exit(failed === 0 ? 0 : 1)
