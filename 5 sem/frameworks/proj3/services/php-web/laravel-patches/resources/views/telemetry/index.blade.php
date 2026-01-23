@extends('layouts.app')

@section('content')
<div class="container mx-auto p-4">
  <div class="page-hero glass p-4 mb-3">
    <div class="d-flex flex-wrap align-items-center justify-content-between">
      <div>
        <div class="text-uppercase small text-muted">Telemetry</div>
        <h2 class="mb-1">Фильтры + поиск по ключевым словам</h2>
        <div class="text-secondary">Сортировка, подсветка совпадений, пагинация</div>
      </div>
      <button class="btn mint-btn" id="quickRefresh">Обновить</button>
    </div>
  </div>

  <div class="panel p-4 rounded-md shadow-sm bg-white mb-4 animate">
    <div class="flex gap-3 flex-wrap items-end animate-fade-in">
      <div class="flex flex-col gap-1">
        <label class="text-xs text-slate-500">Быстрый поиск</label>
        <input id="search" placeholder="notes, file" class="input px-3 py-2 border rounded w-56" />
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-xs text-slate-500">Ключевые слова (через пробел)</label>
        <input id="keywords" placeholder="solar error anomaly" class="input px-3 py-2 border rounded w-64" />
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-xs text-slate-500">Дата от</label>
        <input id="from" type="date" class="input px-3 py-2 border rounded" />
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-xs text-slate-500">Дата до</label>
        <input id="to" type="date" class="input px-3 py-2 border rounded" />
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-xs text-slate-500">Строк на страницу</label>
        <select id="per_page" class="input px-3 py-2 border rounded">
          <option value="10">10</option>
          <option value="25" selected>25</option>
          <option value="50">50</option>
        </select>
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-xs text-slate-500">Сортировка</label>
        <div class="flex gap-2">
          <select id="sort_col" class="input px-3 py-2 border rounded w-40">
            <option value="recorded_at">Время</option>
            <option value="voltage">Voltage</option>
            <option value="temp">Temp</option>
            <option value="count">Count</option>
          </select>
          <select id="sort_dir" class="input px-3 py-2 border rounded w-32">
            <option value="desc">По убыванию</option>
            <option value="asc">По возрастанию</option>
          </select>
        </div>
      </div>
      <div class="flex gap-2">
        <button id="apply" class="btn mint-btn" style="padding: 10px 20px;">Применить</button>
        <button id="reset" class="btn" style="padding: 10px 20px; background: rgba(255,255,255,0.9); color: var(--mint-dark); border: 2px solid var(--mint-medium); border-radius: 12px; font-weight: 600;">Сброс</button>
      </div>
    </div>
  </div>

  <div id="tableWrap" class="rounded-md overflow-auto shadow-sm bg-white animate">
    <table id="teleTable" class="min-w-full text-sm">
      <thead class="bg-gray-50 sticky top-0 shadow-sm">
        <tr>
          <th class="p-2 text-left sortable" data-col="recorded_at">Time ▲▼</th>
          <th class="p-2 text-left sortable" data-col="voltage">Voltage ▲▼</th>
          <th class="p-2 text-left sortable" data-col="temp">Temp ▲▼</th>
          <th class="p-2 text-left sortable" data-col="count">Count ▲▼</th>
          <th class="p-2 text-left">Flags</th>
          <th class="p-2 text-left">Note</th>
          <th class="p-2 text-left">File</th>
        </tr>
      </thead>
      <tbody id="tbody" class="transition-all duration-300">
      </tbody>
    </table>
  </div>

  <div id="pager" class="mt-4"></div>
</div>

<style>
  .highlight { 
    background: linear-gradient(90deg, rgba(168,230,207,0.4), rgba(125,211,192,0.3));
    border-left: 4px solid var(--mint-deep);
    padding-left: 8px;
  }
  .sortable { 
    cursor: pointer; 
    user-select: none;
    transition: all .2s ease;
  }
  .sortable:hover {
    background: rgba(168,230,207,0.2);
    color: var(--mint-dark);
  }
  tr.fade-in { animation: fadeIn 250ms ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
  .flag-true { color: var(--mint-dark); font-weight:600; }
  .flag-false { color: var(--text-dim); }
  .input { 
    border: 2px solid rgba(125,211,192,0.4);
    border-radius: 10px;
    background: rgba(255,255,255,0.9);
    color: var(--text-main);
    transition: all .2s ease;
  }
  .input:focus {
    border-color: var(--mint-deep);
    box-shadow: 0 0 0 4px rgba(95,179,168,0.2);
    outline: none;
    background: rgba(255,255,255,1);
  }
  .btn { transition: transform .12s ease; }
  .btn:active { transform: translateY(1px); }
  .animate { animation: fadeIn .35s ease; }
  .panel {
    background: rgba(255,255,255,0.85);
    border: 2px solid rgba(125,211,192,0.3);
    border-radius: 16px;
    backdrop-filter: blur(10px);
  }
  /* плавное появление карточек/панелей */
  .animate-fade-in { animation: fadeInUp 0.45s ease; }
  @keyframes fadeInUp { from { opacity:0; transform: translateY(10px);} to {opacity:1; transform:none;} }
  #tableWrap {
    background: rgba(255,255,255,0.9);
    border: 2px solid rgba(125,211,192,0.3);
    border-radius: 16px;
  }
  #teleTable thead {
    background: linear-gradient(135deg, var(--mint-medium) 0%, var(--mint-deep) 100%);
    color: var(--text-light);
  }
  #teleTable tbody tr:hover {
    background: rgba(168,230,207,0.2);
  }
  .badge { display:inline-block; padding:2px 6px; border-radius:8px; background:rgba(168,230,207,0.3); color:var(--mint-dark); font-size:11px; }
  mark {
    background: linear-gradient(135deg, rgba(168,230,207,0.5), rgba(125,211,192,0.4));
    color: var(--mint-darker);
    padding: 2px 4px;
    border-radius: 4px;
    font-weight: 600;
  }
  #pager button {
    padding: 8px 16px;
    margin: 0 4px;
    border-radius: 10px;
    border: 2px solid var(--mint-medium);
    background: rgba(255,255,255,0.9);
    color: var(--mint-dark);
    font-weight: 600;
    transition: all .2s ease;
  }
  #pager button:hover:not(:disabled) {
    background: var(--mint-medium);
    color: var(--text-light);
    transform: translateY(-2px);
  }
  #pager button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  #pager span {
    color: var(--text-main);
    font-weight: 500;
    padding: 0 8px;
  }
</style>

<script>
document.addEventListener('DOMContentLoaded', () => {
  let sortCol = 'recorded_at', sortDir = 'desc';
  const tbody = document.getElementById('tbody');
  const pager = document.getElementById('pager');

  function highlight(text, words) {
    if (!words.length || !text) return text || '';
    let t = text;
    words.forEach(w => {
      const re = new RegExp(`(${w})`, 'ig');
      t = t.replace(re, '<mark>$1</mark>');
    });
    return t;
  }

  function isoDateLocal(d){
    const date = new Date(d);
    return date.toLocaleString();
  }

  async function load(page = 1) {
    const params = new URLSearchParams();
    params.set('page', page);
    params.set('per_page', document.getElementById('per_page').value || 25);
    sortCol = document.getElementById('sort_col').value || sortCol;
    sortDir = document.getElementById('sort_dir').value || sortDir;
    params.set('sort', sortCol);
    params.set('dir', sortDir);

    const search = document.getElementById('search').value.trim();
    if (search) params.set('search', search);
    const keywords = (document.getElementById('keywords').value || '').trim();
    if (keywords) params.set('keywords', keywords);

    const from = document.getElementById('from').value;
    if (from) params.set('from', from + ' 00:00:00');

    const to = document.getElementById('to').value;
    if (to) params.set('to', to + ' 23:59:59');

    const res = await fetch(`/api/telemetry?${params.toString()}`);
    const json = await res.json();

    tbody.innerHTML = '';
    const kw = (keywords || '').split(/\s+/).filter(Boolean);
    for (const row of json.data) {
      const tr = document.createElement('tr');
      tr.className = 'fade-in';
      tr.innerHTML = `
        <td class="p-2">${isoDateLocal(row.recorded_at)}</td>
        <td class="p-2">${row.voltage ?? ''}</td>
        <td class="p-2">${row.temp ?? ''}</td>
        <td class="p-2">${row.count ?? ''}</td>
        <td class="p-2">
          <span class="${row.flag_a ? 'flag-true' : 'flag-false'}">A:${row.flag_a ? 'T' : 'F'}</span>
          &nbsp;
          <span class="${row.flag_b ? 'flag-true' : 'flag-false'}">B:${row.flag_b ? 'T' : 'F'}</span>
        </td>
        <td class="p-2">${highlight(row.note ?? '', kw)}</td>
        <td class="p-2">${highlight(row.source_file ?? '', kw)}</td>
      `;
      tbody.appendChild(tr);
    }

    pager.innerHTML = '';
    const prev = document.createElement('button');
    prev.textContent = 'Prev';
    prev.disabled = !json.prev_page_url;
    prev.onclick = () => load(json.current_page - 1);
    pager.appendChild(prev);

    const info = document.createElement('span');
    info.textContent = ` Page ${json.current_page} / ${json.last_page} — ${json.total} rows `;
    pager.appendChild(info);

    const next = document.createElement('button');
    next.textContent = 'Next';
    next.disabled = !json.next_page_url;
    next.onclick = () => load(json.current_page + 1);
    pager.appendChild(next);
  }

  document.querySelectorAll('.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const c = th.dataset.col;
      if (sortCol === c) {
        sortDir = sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        sortCol = c; sortDir = 'desc';
      }
      document.getElementById('sort_col').value = sortCol;
      document.getElementById('sort_dir').value = sortDir;
      load(1);
    });
  });

  document.getElementById('apply').addEventListener('click', () => load(1));
  document.getElementById('reset').addEventListener('click', () => {
    document.getElementById('search').value = '';
    document.getElementById('keywords').value = '';
    document.getElementById('from').value = '';
    document.getElementById('to').value = '';
    document.getElementById('per_page').value = '25';
    document.getElementById('sort_col').value = 'recorded_at';
    document.getElementById('sort_dir').value = 'desc';
    sortCol = 'recorded_at'; sortDir = 'desc';
    load(1);
  });
  document.getElementById('quickRefresh').addEventListener('click', () => load(1));

  let timer;
  document.getElementById('search').addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(() => load(1), 400);
  });

  load(1);
});
</script>
@endsection
