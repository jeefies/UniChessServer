/* 公网访问口令门禁（与服务端 access_auth.require_access_token 对应）。
 *
 * 静态页面本身不设门禁——浏览器打开文档时带不了自定义请求头——所以由这里给
 * 所有 fetch 注入 X-Access-Token；收到 401 就弹遮罩索要口令，拿到后自动重试。
 *
 * 三个页面（index / arena / arena_batch）共用本文件，各自无需改自己的 api()：
 * 本文件包住 window.fetch，页面随后调用的裸 fetch 自然走到这里。
 * arena_batch.html 另有一个管理员口令（X-Admin-Token），它仍由页面自己带，
 * 与这里互不干扰（两道门禁彼此独立）。
 */
(function () {
  'use strict';

  var STORAGE_KEY = 'unichess-access-token';
  var MAX_ATTEMPTS = 3;              // 同一个请求最多重试次数，防死循环

  var nativeFetch = window.fetch && window.fetch.bind(window);
  if (!nativeFetch) return;          // 老浏览器没有 fetch 就整体跳过

  injectStyle();

  function readToken() {
    try { return window.localStorage.getItem(STORAGE_KEY) || ''; } catch (e) { return ''; }
  }
  function saveToken(v) {
    try { window.localStorage.setItem(STORAGE_KEY, v); } catch (e) {}
  }
  function dropToken() {
    try { window.localStorage.removeItem(STORAGE_KEY); } catch (e) {}
  }

  function withToken(headers) {
    var t = readToken();
    if (t) { try { headers.set('X-Access-Token', t); } catch (e) {} }
    return headers;
  }

  // ---- 遮罩 ---------------------------------------------------------------
  var gate = null;                   // { wrap, hint, form, input, error }
  var pending = null;                // { promise, resolve, reject }

  function buildGate() {
    var wrap = document.createElement('div');
    wrap.id = 'access-gate';
    wrap.hidden = true;
    wrap.innerHTML =
      '<div class="ag-card">' +
        '<h2>需要访问口令</h2>' +
        '<p class="ag-hint" id="ag-hint">这个服务的接口需要访问口令，请联系管理员索取。</p>' +
        '<form id="ag-form" autocomplete="off">' +
          '<input id="ag-token" type="password" autocomplete="off" placeholder="访问口令">' +
          '<button id="ag-submit" type="submit">进入</button>' +
        '</form>' +
        '<p class="ag-error" id="ag-error" hidden></p>' +
      '</div>';
    document.body.appendChild(wrap);
    gate = {
      wrap: wrap,
      hint: wrap.querySelector('#ag-hint'),
      form: wrap.querySelector('#ag-form'),
      input: wrap.querySelector('#ag-token'),
      error: wrap.querySelector('#ag-error')
    };
    gate.form.addEventListener('submit', function (ev) {
      ev.preventDefault();
      submitToken();
    });
  }

  function showGate(hint, errorText) {
    if (!gate) buildGate();
    if (hint) gate.hint.textContent = hint;
    gate.error.hidden = !errorText;
    if (errorText) gate.error.textContent = errorText;
    gate.wrap.hidden = false;
    gate.input.value = '';
    gate.input.focus();
  }

  function hideGate() {
    if (gate) gate.wrap.hidden = true;
  }

  function submitToken() {
    var token = (gate.input.value || '').trim();
    if (!token) {
      showGate('口令不能为空。');
      return;
    }
    var p = pending;
    pending = null;
    saveToken(token);
    hideGate();
    p.resolve(token);
  }

  /* 索要口令：并发请求共用同一个遮罩，谁先提交谁放行其余请求。 */
  function ask(hint, errorText) {
    if (!pending) {
      var resolveOuter;
      var rejectOuter;
      var promise = new Promise(function (res, rej) { resolveOuter = res; rejectOuter = rej; });
      pending = { promise: promise, resolve: resolveOuter, reject: rejectOuter };
    }
    showGate(hint, errorText);
    return pending.promise;
  }

  // ---- fetch 包装 ---------------------------------------------------------
  /* 注意：重发依赖 body 是可复用的字符串（三个页面都是 JSON.stringify 的结果）。
   * 若换成 ReadableStream，第二次发送会拿到空 body。 */
  window.fetch = function (input, init) {
    var opts = init ? Object.assign({}, init) : {};
    opts.headers = withToken(new Headers(opts.headers || {}));
    return send(input, opts, 1);
  };

  function send(input, opts, attempt) {
    return nativeFetch(input, opts).then(function (response) {
      if (response.status !== 401) return response;
      if (attempt > MAX_ATTEMPTS) {
        dropToken();
        return response;             // 交给调用方当普通错误处理
      }
      var retrying = attempt > 1;
      return ask(
        retrying ? '上一个口令被拒绝了，请重新输入。' : '请输入访问口令后继续。',
        retrying ? '口令不正确。' : null
      ).then(function () {
        opts.headers = withToken(new Headers(opts.headers || {}));
        return send(input, opts, attempt + 1);
      });
    });
  }

  function injectStyle() {
    var css = document.createElement('style');
    css.textContent =
      '#access-gate{position:fixed;inset:0;z-index:99999;display:flex;align-items:center;' +
        'justify-content:center;background:rgba(6,8,14,.86);font-family:inherit}' +
      '#access-gate[hidden]{display:none}' +
      '#access-gate .ag-card{background:#181b24;border:1px solid rgba(255,255,255,.1);' +
        'border-radius:14px;padding:26px 28px;width:min(420px,92vw);color:#f0f3f8;' +
        'box-shadow:0 24px 60px rgba(0,0,0,.55)}' +
      '#access-gate h2{margin:0 0 8px;font-size:18px;font-weight:600}' +
      '#access-gate .ag-hint{margin:0 0 16px;font-size:13px;color:#9aa3b5;line-height:1.6}' +
      '#access-gate form{display:flex;gap:8px}' +
      '#access-gate input{flex:1;min-width:0;background:#0f1117;' +
        'border:1px solid rgba(255,255,255,.16);border-radius:8px;padding:10px 12px;' +
        'color:#f0f3f8;font-size:14px}' +
      '#access-gate input:focus{outline:none;border-color:#4f6bdf}' +
      '#access-gate button{background:#4f6bdf;border:0;border-radius:8px;padding:10px 16px;' +
        'color:#fff;font-size:14px;cursor:pointer}' +
      '#access-gate button:hover{background:#5f7bea}' +
      '#access-gate .ag-error{margin:12px 0 0;font-size:12px;color:#ff8080}';
    (document.head || document.documentElement).appendChild(css);
  }
})();
