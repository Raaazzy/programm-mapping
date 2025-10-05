export function prepareVisIframeHtml(rawHtml, baseHref) {
  const base = (baseHref || '').replace(/\/+$/, '') + '/'

  const hasHead = /<head[^>]*>/i.test(rawHtml)
  const hasBody = /<body[^>]*>/i.test(rawHtml)
  const hasBase = /<base\s/i.test(rawHtml)
  const hasViewport = /<meta[^>]+name=["']viewport["']/i.test(rawHtml)

  const forceCss = `
  <style id="__embed_vis_full">
    /* базис */
    html, body { margin:0; padding:0; width:100%; height:100vh; }

    .container, .container-fluid { max-width:none !important; width:100% !important; padding:0 !important; margin:0 !important; }
    .row { margin:0 !important; }
    [class^="col"], [class*=" col"] { flex:0 0 100% !important; max-width:100% !important; padding:0 !important; }

    .card, .card-body { width:100% !important; height:100vh !important; margin:0 !important; padding:0 !important; border:0 !important; box-shadow:none !important; background:#fff !important; }

    #__embed_full { position:fixed; inset:0; overflow:auto; background:#fff; height:100vh !important; }

    #mynetwork, svg, canvas {
      width:100% !important; height:100vh !important; display:block;
      box-sizing:border-box;
    }

    hr, .card-header, .card-footer, .navbar { display:none !important; border:0 !important; }
  </style>
`.trim()

  const fitScript = `
  <script id="__embed_vis_fit">
    (function () {
      function tryFit() {
        try {
          if (window.network) {
            if (typeof window.network.setSize === 'function') {
              window.network.setSize('100%', '100%');
            }
            if (typeof window.network.redraw === 'function') {
              window.network.redraw();
            }
            requestAnimationFrame(function () {
              try { window.network.fit({ animation: false }); } catch(e) {}
            });
          }
        } catch (e) {}
      }
      window.addEventListener('load', function () {
        setTimeout(tryFit, 0);
      });
      window.addEventListener('resize', tryFit);
    })();
  </script>`.trim()

  const headInject = [
    !hasBase ? `<base href="${base}">` : '',
    !hasViewport ? `<meta name="viewport" content="width=device-width, initial-scale=1">` : '',
    forceCss,
    fitScript,
  ]
    .filter(Boolean)
    .join('\n')

  let html = rawHtml

  html = html
    .replace(/var\s+network\s*=\s*new\s+vis\.Network/, 'window.network = new vis.Network')
    .replace(/(^|\s)network\s*=\s*new\s+vis\.Network/, ' window.network = new vis.Network')

  html = html.replace(
    /(<div[^>]+id=["']mynetwork["'][^>]*style=["'][^"']*)(width\s*:\s*\d+px;?|height\s*:\s*\d+px;?)([^"']*["'])/gi,
    (m, a, _b, c) => `${a}${c}`,
  )

  if (hasHead) {
    html = html.replace(/<head([^>]*)>/i, `<head$1>\n${headInject}\n`)
  } else {
    if (hasBody) {
      html = html.replace(/<body([^>]*)>/i, `<head>\n${headInject}\n</head><body$1>`)
    } else {
      html = `<!doctype html><html><head>${headInject}</head><body>${rawHtml}</body></html>`
    }
  }
  if (hasBody && !/#__embed_full/.test(html)) {
    html = html
      .replace(
        /<body([^>]*)>/i,
        `<body$1><div id="__embed_full" style="position:fixed;inset:0;overflow:auto;background:#fff;">`,
      )
      .replace(/<\/body>/i, `</div></body>`)
  }

  return html
}
