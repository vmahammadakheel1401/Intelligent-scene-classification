/* Photo organizer: sorts up to 200 photos into scene folders and downloads them as a zip.
 *
 * Flow: pick a folder or photos -> each photo is shrunk in the browser (the model only needs
 * 224 px) -> sent 8 at a time to /api/predict-batch -> grouped into folders -> the user can move
 * photos -> the zip is built in the browser from the ORIGINAL files (nothing is stored on the server).
 */
(function () {
    'use strict';

    var root = document.getElementById('organizer');
    if (!root) return;

    var API = root.dataset.api || '/api/predict-batch';
    var BATCH = parseInt(root.dataset.batch, 10) || 8;
    var MAX = parseInt(root.dataset.max, 10) || 200;
    var LOW = parseFloat(root.dataset.low) || 0.5;
    var READY = root.dataset.ready !== 'no';
    var SHRINK_PX = 384;      // size of the copy sent to the server (and used for thumbnails)
    var PARALLEL = 2;         // batches in flight at the same time
    var ALLOWED = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif'];
    var HEIC = ['heic', 'heif'];

    var FOLDERS = [
        { id: 'mountain', name: 'Mountain' },
        { id: 'sea', name: 'Sea' },
        { id: 'forest', name: 'Forest' },
        { id: 'glacier', name: 'Glacier' },
        { id: 'buildings', name: 'Buildings' },
        { id: 'street', name: 'Street' },
        { id: 'check', name: 'Check these', hint: 'Mixed, unusual or unsure photos. Have a look and move them if needed.' },
        { id: 'not-a-scene', name: 'Not a scene', hint: 'Objects, people, food, screenshots and other photos that are none of the six scenes.' }
    ];
    var FOLDER_BY_ID = {};
    FOLDERS.forEach(function (f) { FOLDER_BY_ID[f.id] = f; });

    var $ = function (id) { return document.getElementById(id); };
    var pick = $('org-pick'), drop = $('org-drop'), folderInput = $('org-folder'), filesInput = $('org-files');
    var progress = $('org-progress'), progressText = $('org-progress-text'), bar = $('org-bar'), barFill = $('org-bar-fill');
    var stopBtn = $('org-stop');
    var results = $('org-results'), summary = $('org-summary'), skippedBox = $('org-skipped');
    var foldersBox = $('org-folders'), announce = $('org-announce');
    var downloadBtn = $('org-download'), restartBtn = $('org-restart');

    var items = [];      // { file, path, name, thumb, result, folder, note }
    var skipped = [];    // { name, why }
    var stopRequested = false;
    var running = false;
    var startedAt = 0;

    if (!READY) {
        folderInput.disabled = true; filesInput.disabled = true;
        drop.classList.add('is-disabled');
    }

    function capitalize(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }
    function ext(name) { var m = /\.([^.]+)$/.exec(name || ''); return m ? m[1].toLowerCase() : ''; }
    function pct(p) { return Math.round(p * 100) + '%'; }

    /* ---------- 1. Collect files (inputs, dropped files and dropped folders) ---------- */

    function readDroppedEntries(dataTransfer) {
        var entries = [];
        if (dataTransfer.items && dataTransfer.items.length && dataTransfer.items[0].webkitGetAsEntry) {
            for (var i = 0; i < dataTransfer.items.length; i++) {
                var entry = dataTransfer.items[i].webkitGetAsEntry();
                if (entry) entries.push(entry);
            }
        }
        if (!entries.length) {  // older browsers: plain files only
            return Promise.resolve(Array.prototype.map.call(dataTransfer.files || [], function (f) { return { file: f, path: f.name }; }));
        }
        var out = [];
        function walk(entry, prefix) {
            if (entry.isFile) {
                return new Promise(function (resolve) {
                    entry.file(function (f) { out.push({ file: f, path: prefix + f.name }); resolve(); }, function () { resolve(); });
                });
            }
            if (entry.isDirectory) {
                var reader = entry.createReader();
                return new Promise(function (resolve) {
                    var all = [];
                    (function readMore() {  // readEntries returns results in chunks
                        reader.readEntries(function (batch) {
                            if (!batch.length) {
                                Promise.all(all.map(function (e) { return walk(e, prefix + entry.name + '/'); })).then(resolve);
                            } else {
                                all = all.concat(Array.prototype.slice.call(batch));
                                readMore();
                            }
                        }, function () { resolve(); });
                    })();
                });
            }
            return Promise.resolve();
        }
        return Promise.all(entries.map(function (e) { return walk(e, ''); })).then(function () { return out; });
    }

    function fromInput(input) {
        return Array.prototype.map.call(input.files || [], function (f) {
            return { file: f, path: f.webkitRelativePath || f.name };
        });
    }

    function start(list) {
        if (running || !READY) return;
        skipped = [];
        var photos = [], ignored = 0;
        list.sort(function (a, b) { return a.path.localeCompare(b.path, undefined, { numeric: true }); });
        list.forEach(function (x) {
            var e = ext(x.file.name);
            if (x.file.name.charAt(0) === '.') return;  // hidden system files
            if (HEIC.indexOf(e) !== -1) {
                skipped.push({ name: x.path, why: 'iPhone HEIC photo: most browsers cannot open it. Convert it to JPG first.' });
            } else if (ALLOWED.indexOf(e) !== -1 || (x.file.type && x.file.type.indexOf('image/') === 0 && e !== 'svg')) {
                photos.push(x);
            } else {
                ignored++;
            }
        });
        if (photos.length > MAX) {
            photos.slice(MAX).forEach(function (x) { skipped.push({ name: x.path, why: 'Over the ' + MAX + '-photo limit for one run. Sort it in a second run.' }); });
            photos = photos.slice(0, MAX);
        }
        if (!photos.length) {
            showPickMessage(ignored || skipped.length
                ? 'No photos we can sort were found' + (skipped.length ? ' (' + skipped.length + ' HEIC photos need converting to JPG first).' : '.')
                : 'That folder is empty.');
            return;
        }
        items = photos.map(function (x) {
            return { file: x.file, path: x.path, name: x.file.name, thumb: '', result: null, folder: null, note: '' };
        });
        run(ignored);
    }

    function showPickMessage(text) {
        var p = pick.querySelector('.org-pick-error');
        if (!p) {
            p = document.createElement('p');
            p.className = 'form-error org-pick-error';
            p.setAttribute('role', 'alert');
            pick.appendChild(p);
        }
        p.textContent = text;
    }

    /* ---------- 2. Shrink and classify ---------- */

    function shrink(file) {
        return new Promise(function (resolve) {
            var url = URL.createObjectURL(file);
            var img = new Image();
            img.onload = function () {
                var scale = Math.min(1, SHRINK_PX / Math.max(img.naturalWidth, img.naturalHeight));
                var c = document.createElement('canvas');
                c.width = Math.max(1, Math.round(img.naturalWidth * scale));
                c.height = Math.max(1, Math.round(img.naturalHeight * scale));
                var ctx = c.getContext('2d');
                ctx.fillStyle = '#fff';  // transparent PNGs get a white background, like most viewers show them
                ctx.fillRect(0, 0, c.width, c.height);
                ctx.drawImage(img, 0, 0, c.width, c.height);
                URL.revokeObjectURL(url);
                c.toBlob(function (blob) { resolve(blob); }, 'image/jpeg', 0.85);
            };
            img.onerror = function () { URL.revokeObjectURL(url); resolve(null); };
            img.src = url;
        });
    }

    function postBatch(form, attempt) {
        return fetch(API, { method: 'POST', body: form }).then(function (res) {
            return res.json().then(function (data) {
                if (!res.ok || !data.ok) throw new Error(data.error || 'Server error ' + res.status);
                return data.results;
            });
        }).catch(function (err) {
            if (attempt < 2) return new Promise(function (r) { setTimeout(r, 800); }).then(function () { return postBatch(form, attempt + 1); });
            throw err;
        });
    }

    function classifyBatch(batch) {
        return Promise.all(batch.map(function (it) { return shrink(it.file); })).then(function (blobs) {
            var form = new FormData(), sent = [];
            blobs.forEach(function (blob, i) {
                var it = batch[i];
                if (!blob) { it.error = "This file couldn't be opened as an image."; return; }
                it.thumb = URL.createObjectURL(blob);
                form.append('images', blob, it.name.replace(/\.[^.]*$/, '') + '.jpg');
                sent.push(it);
            });
            if (!sent.length) return;
            return postBatch(form, 0).then(function (list) {
                sent.forEach(function (it, i) {
                    var r = list[i];
                    if (r && r.ok) it.result = r; else it.error = (r && r.error) || 'Could not be classified.';
                });
            }).catch(function () {
                sent.forEach(function (it) { it.error = 'Could not reach the server. Check your connection and try again.'; });
            });
        });
    }

    function setProgress(done, total) {
        var p = total ? Math.round(done / total * 100) : 0;
        barFill.style.width = p + '%';
        bar.setAttribute('aria-valuenow', String(p));
        progressText.textContent = 'Sorting ' + done + ' of ' + total + ' photos…';
    }

    function run(ignored) {
        running = true;
        stopRequested = false;
        startedAt = Date.now();
        pick.hidden = true;
        results.hidden = true;
        progress.hidden = false;
        stopBtn.disabled = false;
        stopBtn.textContent = 'Stop';
        var total = items.length, done = 0, next = 0;
        setProgress(0, total);
        progress.scrollIntoView({ block: 'nearest' });

        function worker() {
            if (stopRequested || next >= total) return Promise.resolve();
            var batch = items.slice(next, next + BATCH);
            next += batch.length;
            return classifyBatch(batch).then(function () {
                done += batch.length;
                setProgress(done, total);
                return worker();
            });
        }
        var workers = [];
        for (var i = 0; i < PARALLEL; i++) workers.push(worker());
        Promise.all(workers).then(function () {
            running = false;
            items.forEach(function (it) {
                if (it.result) {
                    var a = assign(it.result);
                    it.folder = a.folder; it.note = a.note;
                } else {
                    skipped.push({ name: it.path, why: it.error || 'Not sorted: you pressed Stop.' });
                    if (it.thumb) URL.revokeObjectURL(it.thumb);
                }
            });
            items = items.filter(function (it) { return it.result; });
            showResults(ignored);
        });
    }

    // Which folder a result goes into, and the short note shown under the photo.
    function assign(r) {
        var guess = capitalize(r.label) + ' ' + pct(r.confidence);
        if (r.in_scope === false) return { folder: 'not-a-scene', note: "Doesn't look like a scene" };
        if (r.comparison) return { folder: 'check', note: r.comparison.title || ('Close call: ' + guess) };
        if (r.scene_check && r.scene_check.status === 'unusual') return { folder: 'check', note: 'Unusual photo · best guess ' + guess };
        if (r.confidence < LOW) return { folder: 'check', note: 'Not sure · best guess ' + guess };
        return { folder: r.label, note: pct(r.confidence) + ' confident' };
    }

    /* ---------- 3. Show and edit the folders ---------- */

    function counts() {
        var c = {};
        FOLDERS.forEach(function (f) { c[f.id] = 0; });
        items.forEach(function (it) { c[it.folder] = (c[it.folder] || 0) + 1; });
        return c;
    }

    function showResults(ignored) {
        progress.hidden = true;
        results.hidden = false;
        var secs = Math.max(1, Math.round((Date.now() - startedAt) / 1000));
        var c = counts();
        var parts = FOLDERS.filter(function (f) { return c[f.id]; }).map(function (f) { return c[f.id] + ' ' + f.name.toLowerCase(); });
        summary.textContent = items.length
            ? 'Sorted ' + items.length + ' photo' + (items.length === 1 ? '' : 's') + ' in ' + secs + ' s: ' + parts.join(', ') + '.'
            : 'No photos were sorted.';
        downloadBtn.disabled = !items.length;

        skippedBox.hidden = !skipped.length && !ignored;
        skippedBox.innerHTML = '';
        if (skipped.length || ignored) {
            var body = document.createElement('div');
            var head = document.createElement('p');
            head.innerHTML = '<strong></strong>';
            head.firstChild.textContent = (skipped.length ? skipped.length + ' photo' + (skipped.length === 1 ? ' was' : 's were') + ' not sorted.' : '') +
                (ignored ? ' ' + ignored + ' non-photo file' + (ignored === 1 ? ' was' : 's were') + ' ignored.' : '');
            body.appendChild(head);
            if (skipped.length) {
                var det = document.createElement('details');
                var sum = document.createElement('summary');
                sum.textContent = 'Show which ones';
                det.appendChild(sum);
                var ul = document.createElement('ul');
                skipped.slice(0, 50).forEach(function (s) {
                    var li = document.createElement('li');
                    li.textContent = s.name + ': ' + s.why;
                    ul.appendChild(li);
                });
                if (skipped.length > 50) { var more = document.createElement('li'); more.textContent = '…and ' + (skipped.length - 50) + ' more.'; ul.appendChild(more); }
                det.appendChild(ul);
                body.appendChild(det);
            }
            skippedBox.appendChild(body);
        }
        renderFolders();
        announce.textContent = summary.textContent;
        results.scrollIntoView({ block: 'start' });
    }

    function renderFolders() {
        var c = counts();
        foldersBox.innerHTML = '';
        FOLDERS.forEach(function (f) {
            if (!c[f.id]) return;
            var sec = document.createElement('section');
            sec.className = 'org-folder org-folder-' + f.id;
            sec.setAttribute('aria-labelledby', 'folder-' + f.id);

            var head = document.createElement('div');
            head.className = 'org-folder-head';
            var h = document.createElement('h3');
            h.id = 'folder-' + f.id;
            h.innerHTML = '<span class="org-folder-name"></span> <span class="org-count num"></span>';
            h.querySelector('.org-folder-name').textContent = f.name;
            h.querySelector('.org-count').textContent = c[f.id];
            head.appendChild(h);
            var dl = document.createElement('button');
            dl.type = 'button';
            dl.className = 'btn btn-quiet btn-small';
            dl.textContent = 'Download this folder';
            dl.setAttribute('aria-label', 'Download the ' + f.name + ' folder as a zip');
            dl.addEventListener('click', function () { downloadZip([f.id], dl); });
            head.appendChild(dl);
            sec.appendChild(head);
            if (f.hint) {
                var hint = document.createElement('p');
                hint.className = 'org-folder-hint';
                hint.textContent = f.hint;
                sec.appendChild(hint);
            }

            var grid = document.createElement('ul');
            grid.className = 'org-grid';
            items.forEach(function (it, idx) {
                if (it.folder !== f.id) return;
                grid.appendChild(tile(it, idx));
            });
            sec.appendChild(grid);
            foldersBox.appendChild(sec);
        });
    }

    function tile(it, idx) {
        var li = document.createElement('li');
        li.className = 'org-tile';
        var img = document.createElement('img');
        img.src = it.thumb;
        img.alt = '';
        img.loading = 'lazy';
        img.decoding = 'async';
        li.appendChild(img);
        var name = document.createElement('p');
        name.className = 'org-tile-name';
        name.textContent = it.name;
        name.title = it.path;
        li.appendChild(name);
        var note = document.createElement('p');
        note.className = 'org-tile-note';
        note.textContent = it.moved ? 'Moved by you' : it.note;
        li.appendChild(note);

        var label = document.createElement('label');
        label.className = 'org-move';
        var span = document.createElement('span');
        span.textContent = 'Move to';
        label.appendChild(span);
        var sel = document.createElement('select');
        sel.setAttribute('aria-label', 'Move ' + it.name + ' to another folder');
        FOLDERS.forEach(function (f) {
            var o = document.createElement('option');
            o.value = f.id;
            o.textContent = f.name;
            if (f.id === it.folder) o.selected = true;
            sel.appendChild(o);
        });
        sel.addEventListener('change', function () {
            it.folder = sel.value;
            it.moved = true;
            var y = window.scrollY;
            renderFolders();
            window.scrollTo(0, y);
            announce.textContent = it.name + ' moved to ' + FOLDER_BY_ID[it.folder].name + '.';
            var again = foldersBox.querySelector('[data-idx="' + idx + '"] select');
            if (again) again.focus();
        });
        label.appendChild(sel);
        li.appendChild(label);
        li.dataset.idx = String(idx);
        return li;
    }

    /* ---------- 4. Zip download (original files, built in the browser) ---------- */

    function csvCell(v) {
        v = String(v == null ? '' : v);
        return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
    }

    function downloadZip(folderIds, btn) {
        if (!window.JSZip) {
            alert('The zip tool could not be loaded. Check your internet connection and reload the page.');
            return;
        }
        var zip = new window.JSZip();
        var used = {};
        var rows = [['photo', 'original_path', 'folder', 'best_guess', 'confidence', 'note']];
        items.forEach(function (it) {
            if (folderIds.indexOf(it.folder) === -1) return;
            var folder = FOLDER_BY_ID[it.folder].name;
            var name = it.name, base = name.replace(/(\.[^.]*)?$/, ''), dot = name.slice(base.length), n = 2;
            used[folder] = used[folder] || {};
            while (used[folder][name.toLowerCase()]) { name = base + ' (' + n++ + ')' + dot; }
            used[folder][name.toLowerCase()] = true;
            zip.file(folder + '/' + name, it.file, { binary: true });
            rows.push([folder + '/' + name, it.path, folder, it.result.label, it.result.confidence, it.moved ? 'moved by you' : it.note]);
        });
        zip.file('report.csv', rows.map(function (r) { return r.map(csvCell).join(','); }).join('\n'));

        var labelEl = btn.querySelector('.btn-label') || btn;
        var oldLabel = labelEl.textContent;
        btn.disabled = true;
        btn.parentNode.classList.add('is-busy');
        zip.generateAsync({ type: 'blob', compression: 'STORE', streamFiles: true }, function (meta) {
            labelEl.textContent = 'Making zip… ' + Math.round(meta.percent) + '%';
        }).then(function (blob) {
            var a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = folderIds.length === 1 ? 'photos_' + FOLDER_BY_ID[folderIds[0]].name.toLowerCase().replace(/ /g, '_') + '.zip' : 'sorted_photos.zip';
            document.body.appendChild(a);
            a.click();
            setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 4000);
        }).catch(function () {
            alert('The zip could not be made, possibly because the photos are too large for this device. Try downloading one folder at a time.');
        }).then(function () {
            labelEl.textContent = oldLabel;
            btn.disabled = false;
            btn.parentNode.classList.remove('is-busy');
        });
    }

    /* ---------- Events ---------- */

    folderInput.addEventListener('change', function () { start(fromInput(folderInput)); folderInput.value = ''; });
    filesInput.addEventListener('change', function () { start(fromInput(filesInput)); filesInput.value = ''; });

    ['dragenter', 'dragover'].forEach(function (t) {
        drop.addEventListener(t, function (e) { e.preventDefault(); if (READY) drop.classList.add('is-dragover'); });
    });
    ['dragleave', 'dragend', 'drop'].forEach(function (t) {
        drop.addEventListener(t, function () { drop.classList.remove('is-dragover'); });
    });
    drop.addEventListener('drop', function (e) {
        e.preventDefault();
        if (!READY || running) return;
        readDroppedEntries(e.dataTransfer).then(start);
    });
    // A photo dropped next to the box should not open in the tab.
    window.addEventListener('dragover', function (e) { e.preventDefault(); });
    window.addEventListener('drop', function (e) { e.preventDefault(); });

    stopBtn.addEventListener('click', function () {
        stopRequested = true;
        stopBtn.disabled = true;
        stopBtn.textContent = 'Stopping…';
    });

    downloadBtn.addEventListener('click', function () {
        downloadZip(FOLDERS.map(function (f) { return f.id; }), downloadBtn);
    });

    restartBtn.addEventListener('click', function () {
        items.forEach(function (it) { if (it.thumb) URL.revokeObjectURL(it.thumb); });
        items = []; skipped = [];
        results.hidden = true;
        foldersBox.innerHTML = '';
        pick.hidden = false;
        var err = pick.querySelector('.org-pick-error');
        if (err) err.remove();
        pick.scrollIntoView({ block: 'start' });
    });

    // Leaving mid-sort loses the work: ask first.
    window.addEventListener('beforeunload', function (e) {
        if (running) { e.preventDefault(); e.returnValue = ''; }
    });
})();
