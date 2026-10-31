#!/usr/bin/env node
/**
 * Rice_system 前端部署服务器
 *
 * 作用等价于一段最小 nginx 配置：
 *   /            -> RiceAnylazeWeb/dist（SPA history 回退到 index.html）
 *   /api/...     -> FastAPI 后端
 *   /static/...  -> FastAPI 后端挂载的本地文件目录
 *
 * 因为浏览器只访问本服务器，前后端同源，所以不需要 CORS，
 * 前端也不需要把后端地址打包进产物（VITE_API_BASE_URL=/api）。
 *
 * 用法：
 *   node RiceAnylaze/deploy/runtime/serve.mjs [dist目录]
 * 环境变量：
 *   PORT=5173              监听端口
 *   HOST=0.0.0.0           监听地址（0.0.0.0 允许局域网访问）
 *   BACKEND_ORIGIN=http://127.0.0.1:8000   后端地址
 */
import http from 'node:http'
import { createReadStream, existsSync, statSync } from 'node:fs'
import { extname, join, normalize, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(
  process.argv[2] || process.env.WEB_ROOT || fileURLToPath(new URL('../../../RiceAnylazeWeb/dist/', import.meta.url)),
)
const PORT = Number(process.env.PORT || 5173)
const HOST = process.env.HOST || '0.0.0.0'
const BACKEND = new URL(process.env.BACKEND_ORIGIN || 'http://127.0.0.1:8000')

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.ico': 'image/x-icon',
  '.webp': 'image/webp',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.txt': 'text/plain; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
}

if (!existsSync(ROOT)) {
  console.error(`[serve] 前端产物目录不存在: ${ROOT}\n        请先在 RiceAnylazeWeb 下执行 npm run build`)
  process.exit(1)
}

function serveStatic(req, res, rel) {
  if (rel.endsWith('/')) rel += 'index.html'

  let file = normalize(join(ROOT, rel))
  if (file !== ROOT && !file.startsWith(ROOT + sep)) {
    res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' })
    return res.end('403 Forbidden')
  }

  if (!existsSync(file) || statSync(file).isDirectory()) {
    // SPA history 模式回退：/login、/user-center 等交给前端路由
    if (extname(file)) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' })
      return res.end('404 Not Found')
    }
    file = join(ROOT, 'index.html')
  }

  const stat = statSync(file)
  const type = MIME[extname(file).toLowerCase()] || 'application/octet-stream'
  const immutable = file.includes(`${sep}assets${sep}`)
  res.writeHead(200, {
    'Content-Type': type,
    'Content-Length': stat.size,
    'Cache-Control': immutable ? 'public, max-age=31536000, immutable' : 'no-cache',
  })
  if (req.method === 'HEAD') return res.end()
  createReadStream(file).pipe(res)
}

function proxy(req, res, url) {
  const upstream = http.request(
    {
      hostname: BACKEND.hostname,
      port: BACKEND.port || 80,
      path: url.pathname + url.search,
      method: req.method,
      // 原样转发请求头（含 Host）：后端用 request.base_url 生成的 /static/... 图片地址
      // 才会指回本服务器，保持同源。
      headers: req.headers,
    },
    (up) => {
      res.writeHead(up.statusCode || 502, up.headers)
      up.pipe(res)
    },
  )

  upstream.on('error', (err) => {
    console.error(`[serve] 代理到后端失败: ${req.method} ${url.pathname} -> ${err.message}`)
    if (!res.headersSent) {
      res.writeHead(502, { 'Content-Type': 'application/json; charset=utf-8' })
    }
    res.end(JSON.stringify({ detail: `后端不可用: ${err.message}` }))
  })

  req.pipe(upstream)
}

const server = http.createServer((req, res) => {
  let url
  let pathname
  try {
    url = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`)
    pathname = decodeURIComponent(url.pathname)
  } catch {
    res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' })
    return res.end('400 Bad Request')
  }
  if (url.pathname === '/api' || url.pathname.startsWith('/api/') ||
      url.pathname === '/static' || url.pathname.startsWith('/static/')) {
    return proxy(req, res, url)
  }
  return serveStatic(req, res, pathname)
})

server.listen(PORT, HOST, () => {
  console.log(`[serve] 前端已启动: http://localhost:${PORT}/`)
  console.log(`[serve] 静态目录: ${ROOT}`)
  console.log(`[serve] /api、/static -> ${BACKEND.origin}`)
})
