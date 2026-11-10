import type { Register } from 'claude-code'

type Tool = { name: string; check: string[]; install: string[] }

const TOOLS: Tool[] = [
  { name: 'ffmpeg', check: ['ffmpeg', '-version'], install: ['apt-get', 'install', '-y', 'ffmpeg'] },
  { name: 'whisper', check: ['python3', '-c', 'import whisper'], install: ['pip', 'install', '-q', 'openai-whisper'] },
  { name: 'hyperframes', check: ['npx', '--no-install', 'hyperframes', '--version'], install: ['npm', 'install', 'hyperframes'] },
]

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'media-tools',
      description: 'Check FFmpeg, Whisper and Hyperframes, and install what is missing',
    })
    return next(e)
  })

  on('command.run', { command: 'media-tools' }, async $ => {
    const lines: string[] = []
    for (const t of TOOLS) {
      const ok = async () => (await $.process.run(t.check, { timeoutMs: 60000 }).catch(() => ({ exitCode: 1 }))).exitCode === 0
      if (await ok()) {
        lines.push(`${t.name}: ok`)
        continue
      }
      const r = await $.process.run(t.install, { timeoutMs: 600000 }).catch(err => ({ exitCode: 1, stderr: String(err) }))
      lines.push(`${t.name}: ${r.exitCode === 0 && (await ok()) ? 'installed' : 'install failed'}`)
    }
    return { text: lines.join('\n') }
  })
}
