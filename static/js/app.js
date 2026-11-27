// Classify page: preview, validation, drag and drop, paste, samples, results without reloading, history.
(function () {
    'use strict';

    var form = document.getElementById('upload-form');
    if (!form) return;

    // With JavaScript, big photos are shrunk before upload, so the page can accept larger files
    // than the server limit (data-max-mb, used by the no-JavaScript form).
    var MAX_MB = parseFloat(form.dataset.pickMb) || parseFloat(form.dataset.maxMb) || 10;
    var hint = document.getElementById('file-hint');
    if (hint) hint.textContent = hint.textContent.replace(/up to [\d.]+ MB/, 'up to ' + MAX_MB + ' MB');
    var LOW = parseFloat(form.dataset.low) || 0.5;
    var ALLOWED = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif'];
    var HISTORY_KEY = 'scene-history';
    var HISTORY_MAX = 8;

    var input = document.getElementById('image');
    var dropzone = document.getElementById('dropzone');
    // The preview <img> is created here (not in the HTML) so there is never an empty image tag.
    var preview = document.createElement('img');
    preview.id = 'preview';
    preview.className = 'dropzone-preview';
    preview.alt = '';
    preview.hidden = true;
    dropzone.insertBefore(preview, dropzone.firstChild);
    var emptyState = document.getElementById('dropzone-empty');
    var fileMeta = document.getElementById('file-meta');
    var errorBox = document.getElementById('form-error');
    var errorText = document.getElementById('form-error-text');
    var classifyBtn = document.getElementById('classify-btn');
    var resetBtn = document.getElementById('reset-btn');
    var result = document.getElementById('result');
    var verdict = document.getElementById('verdict');
    var confidence = document.getElementById('confidence');
    var verdictDesc = document.getElementById('verdict-desc');
    var unsure = document.getElementById('notice-unsure');
    var scopeNote = document.getElementById('notice-scope');
    var comparisonNote = document.getElementById('notice-comparison');
    var comparisonTitle = document.getElementById('comparison-title');
    var comparisonScores = document.getElementById('comparison-scores');
    var comparisonGuidance = document.getElementById('comparison-guidance');
    var announce = document.getElementById('result-announce');
    var rows = Array.prototype.slice.call(document.querySelectorAll('#spectrum .scene'));
    var historySection = document.getElementById('history');
    var historyList = document.getElementById('history-list');
    var historyClear = document.getElementById('history-clear');

    var currentFile = null;
    var previewUrl = null;
    var busy = false;

    // We validate ourselves so we can show friendly messages instead of the browser's bubble.
    form.noValidate = true;

    function reducedMotion() { return window.matchMedia('(prefers-reduced-motion: reduce)').matches; }
    function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
    function formatSize(bytes) {
        return bytes < 1024 * 1024 ? Math.max(1, Math.round(bytes / 1024)) + ' KB' : (bytes / 1024 / 1024).toFixed(1) + ' MB';
    }

    function showError(message) {
        errorText.textContent = message;
        errorBox.hidden = false;
    }
    function clearError() { errorBox.hidden = true; errorText.textContent = ''; }

    function validate(file) {
        if (!file) return 'Please choose an image first.';
        var ext = (file.name.split('.').pop() || '').toLowerCase();
        var looksLikeImage = (file.type && file.type.indexOf('image/') === 0) || ALLOWED.indexOf(ext) !== -1;
        if (!looksLikeImage || (file.type && file.type.indexOf('image/') !== 0)) {
            return '"' + file.name + '" isn\'t an image. Please choose a JPG, PNG, WEBP, BMP or GIF photo.';
        }
        if (file.size > MAX_MB * 1024 * 1024) {
            return 'That photo is ' + formatSize(file.size) + '. Please choose one smaller than ' + MAX_MB + ' MB.';
        }
        if (file.size === 0) return 'That file is empty. Please choose another photo.';
        return null;
    }

    function setResultIdle() {
        result.dataset.state = 'idle';
        verdict.textContent = 'Waiting for a photo';
        confidence.textContent = currentFile ? 'Press "Classify photo" to see what the model thinks.' : 'Pick a photo, or try a sample below.';
        verdictDesc.textContent = '';
        unsure.hidden = true;
        comparisonNote.hidden = true;
        scopeNote.hidden = true;
        result.classList.remove('is-out-of-scope');
        rows.forEach(function (row) {
            row.classList.remove('is-top');
            row.style.setProperty('--p', 0);
            row.querySelector('.scene-pct').textContent = '–';
            var rank = row.querySelector('.scene-rank');
            rank.hidden = true;
            rank.textContent = '';
        });
    }

    function setFile(file) {
        var problem = validate(file);
        if (problem) {
            showError(problem);
            return false;
        }
        clearError();
        currentFile = file;
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = URL.createObjectURL(file);
        preview.onload = function () {
            // Let the frame take the photo's own shape (within limits) so the photo fills it.
            var r = preview.naturalWidth / preview.naturalHeight;
            if (r) dropzone.style.setProperty('--img-ratio', Math.min(1.9, Math.max(0.75, r)).toFixed(3));
        };
        preview.src = previewUrl;
        preview.alt = 'Preview of ' + file.name;
        preview.hidden = false;
        emptyState.hidden = true;
        dropzone.classList.add('has-image');
        fileMeta.textContent = file.name + ' · ' + formatSize(file.size);
        resetBtn.hidden = false;
        setResultIdle();
        return true;
    }

    function clearFile() {
        currentFile = null;
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = null;
        preview.removeAttribute('src');
        preview.hidden = true;
        emptyState.hidden = false;
        dropzone.classList.remove('has-image');
        dropzone.style.removeProperty('--img-ratio');
        input.value = '';
        fileMeta.textContent = 'No photo chosen yet.';
        resetBtn.hidden = true;
        clearError();
        markActiveSample(null);
        setResultIdle();
    }

    function setBusy(on) {
        busy = on;
        form.classList.toggle('is-busy', on);
        result.classList.toggle('is-busy', on);
        result.setAttribute('aria-busy', on ? 'true' : 'false');
        classifyBtn.disabled = on;
        classifyBtn.querySelector('.btn-label').textContent = on ? 'Classifying…' : 'Classify photo';
        document.querySelectorAll('.sample').forEach(function (b) { b.disabled = on; });
    }

    var OUT_OF_SCOPE_TEXT = "It doesn't look like buildings, a forest, a glacier, a mountain, the sea or a street. " +
        'It may be an object, a person, a drawing or a screenshot. Try a photo of one of the six scenes.';

    function renderResult(data) {
        result.dataset.state = 'done';
        var pct = Math.round(data.confidence * 100);
        var outOfScope = data.in_scope === false;
        var unusual = !outOfScope && data.scene_check && data.scene_check.status === 'unusual';
        result.classList.toggle('is-out-of-scope', outOfScope);
        confidence.innerHTML = '';
        if (outOfScope) {
            verdict.textContent = 'Not a scene photo';
            confidence.textContent = "Doesn't match any of the six scenes";
            verdictDesc.textContent = OUT_OF_SCOPE_TEXT + ' (If it had to pick, the model would say ' + data.label + ', ' + pct + '%.)';
        } else {
            verdict.textContent = capitalize(data.label);
            var num = document.createElement('span');
            num.className = 'num';
            num.textContent = pct + '%';
            confidence.appendChild(num);
            confidence.appendChild(document.createTextNode(' confident'));
            verdictDesc.textContent = data.description || '';
        }
        var comparison = data.comparison;
        unsure.hidden = !(data.confidence < LOW) || !!comparison || outOfScope;
        scopeNote.hidden = !unusual;
        comparisonNote.hidden = !comparison;
        if (comparison) {
            comparisonTitle.textContent = comparison.title || ('Close call: ' + capitalize(comparison.first_label) + ' or ' + capitalize(comparison.second_label) + '?');
            comparisonScores.textContent =
                capitalize(comparison.first_label) + ' ' + (comparison.first_probability * 100).toFixed(1) + '% · ' +
                capitalize(comparison.second_label) + ' ' + (comparison.second_probability * 100).toFixed(1) + '%';
            comparisonGuidance.textContent = comparison.guidance;
        }

        var probs = {};
        (data.probabilities || data.top3).forEach(function (p) { probs[p.label] = p.probability; });
        var ranks = {};
        data.top3.forEach(function (t, i) { ranks[t.label] = i + 1; });

        rows.forEach(function (row, i) {
            var name = row.dataset.class;
            var p = probs[name] || 0;
            row.querySelector('.bar-fill').style.transitionDelay = (i * 45) + 'ms';
            row.style.setProperty('--p', p);
            row.querySelector('.scene-pct').textContent = (p * 100).toFixed(1) + '%';
            row.classList.toggle('is-top', ranks[name] === 1);
            var rank = row.querySelector('.scene-rank');
            rank.hidden = !ranks[name];
            rank.textContent = ranks[name] ? '#' + ranks[name] : '';
        });

        var second = data.top3[1];
        if (outOfScope) {
            announce.textContent = 'Result: not a scene photo. ' + OUT_OF_SCOPE_TEXT;
            return;
        }
        announce.textContent = 'Result: ' + capitalize(data.label) + ', ' + pct + ' percent confident.' +
            (unusual ? ' This photo looks unusual for the model, so treat the answer with care.' : '') +
            (second ? ' Second choice: ' + second.label + ', ' + Math.round(second.probability * 100) + ' percent.' : '') +
            (comparison ? ' ' + (comparison.title || 'Close call') + '. ' + comparison.guidance : '') +
            (data.confidence < LOW ? ' The model is not sure about this photo.' : '');
    }

    // Big photos are shrunk in the browser before upload: the model only needs 224 px, and
    // Vercel rejects requests over 4.5 MB. Small files are sent as they are.
    var SHRINK_OVER_BYTES = 1.5 * 1024 * 1024;
    var SHRINK_TO_PX = 1024;

    function shrinkForUpload(file) {
        if (file.size <= SHRINK_OVER_BYTES || !window.HTMLCanvasElement) return Promise.resolve(file);
        return new Promise(function (resolve) {
            var url = URL.createObjectURL(file);
            var img = new Image();
            img.onload = function () {
                var scale = Math.min(1, SHRINK_TO_PX / Math.max(img.naturalWidth, img.naturalHeight));
                var canvas = document.createElement('canvas');
                canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
                canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
                canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
                URL.revokeObjectURL(url);
                canvas.toBlob(function (blob) {
                    if (!blob) return resolve(file);
                    var name = file.name.replace(/\.[^.]*$/, '') + '.jpg';
                    resolve(new File([blob], name, { type: 'image/jpeg' }));
                }, 'image/jpeg', 0.9);
            };
            img.onerror = function () { URL.revokeObjectURL(url); resolve(file); };
            img.src = url;
        });
    }

    function classify() {
        if (busy) return;
        if (!currentFile) {
            showError('Please choose an image first.');
            input.focus();
            return;
        }
        clearError();
        setBusy(true);
        shrinkForUpload(currentFile)
            .then(function (upload) {
                var body = new FormData();
                body.append('image', upload, upload.name);
                return fetch(form.dataset.api || '/api/predict', { method: 'POST', body: body });
            })
            .then(function (res) {
                return res.json().catch(function () {
                    throw new Error(res.status === 413 ? 'That photo is too big. Please choose one smaller than ' + MAX_MB + ' MB.' : 'The server sent an unexpected reply. Please try again.');
                });
            })
            .then(function (data) {
                if (!data.ok) throw new Error(data.error || 'Something went wrong. Please try again.');
                renderResult(data);
                saveHistory(data);
                if (window.matchMedia('(max-width: 860px)').matches) {
                    // On phones the result sits below the photo, so bring it into view.
                    result.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'start' });
                }
            })
            .catch(function (err) {
                var msg = err && err.message;
                if (!msg || /fetch|network/i.test(msg)) msg = 'Couldn\'t reach the app. Check that it is still running, then try again.';
                showError(msg);
                setResultIdle();
            })
            .then(function () { setBusy(false); });
    }

    // ----- Events -----
    input.addEventListener('change', function () {
        if (input.files && input.files[0]) {
            markActiveSample(null);
            setFile(input.files[0]);
        }
    });

    form.addEventListener('submit', function (e) {
        e.preventDefault();
        classify();
    });

    resetBtn.addEventListener('click', function () {
        clearFile();
        input.focus();
    });

    ['dragenter', 'dragover'].forEach(function (type) {
        dropzone.addEventListener(type, function (e) {
            e.preventDefault();
            dropzone.classList.add('is-dragover');
        });
    });
    ['dragleave', 'dragend', 'drop'].forEach(function (type) {
        dropzone.addEventListener(type, function () { dropzone.classList.remove('is-dragover'); });
    });
    dropzone.addEventListener('drop', function (e) {
        e.preventDefault();
        var file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        markActiveSample(null);
        if (setFile(file)) classify();
    });
    // Stop a photo dropped next to the box from opening in the browser tab.
    window.addEventListener('dragover', function (e) { e.preventDefault(); });
    window.addEventListener('drop', function (e) { e.preventDefault(); });

    // Paste a copied image with Ctrl+V.
    document.addEventListener('paste', function (e) {
        var items = (e.clipboardData && e.clipboardData.files) || [];
        if (items.length && items[0].type.indexOf('image/') === 0) {
            var f = items[0];
            var named = new File([f], f.name && f.name !== 'image.png' ? f.name : 'pasted-image.png', { type: f.type });
            markActiveSample(null);
            if (setFile(named)) classify();
        }
    });

    // ----- Samples -----
    function markActiveSample(btn) {
        document.querySelectorAll('.sample').forEach(function (b) {
            b.classList.toggle('is-active', b === btn);
            b.setAttribute('aria-pressed', b === btn ? 'true' : 'false');
        });
    }

    document.querySelectorAll('.sample').forEach(function (btn) {
        btn.addEventListener('click', function () {
            if (busy) return;
            markActiveSample(btn);
            fetch(btn.dataset.src)
                .then(function (res) {
                    if (!res.ok) throw new Error();
                    return res.blob();
                })
                .then(function (blob) {
                    var file = new File([blob], btn.dataset.name, { type: blob.type || 'image/jpeg' });
                    if (setFile(file)) {
                        classify();
                        if (btn.dataset.auto) { delete btn.dataset.auto; return; }  // demo link: stay at the top
                        form.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'start' });
                    }
                })
                .catch(function () { showError('Couldn\'t load that sample photo. Please try another one.'); });
        });
    });

    // ----- History (saved only in this browser) -----
    function readHistory() {
        try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]') || []; } catch (e) { return []; }
    }
    function writeHistory(list) {
        try { localStorage.setItem(HISTORY_KEY, JSON.stringify(list)); return true; } catch (e) { return false; }
    }

    function makeThumb() {
        // Small JPEG of the preview so history stays tiny in storage.
        try {
            var w = 160, h = 120;
            var c = document.createElement('canvas');
            c.width = w; c.height = h;
            var ctx = c.getContext('2d');
            var iw = preview.naturalWidth, ih = preview.naturalHeight;
            if (!iw || !ih) return '';
            var scale = Math.max(w / iw, h / ih);
            var dw = iw * scale, dh = ih * scale;
            ctx.drawImage(preview, (w - dw) / 2, (h - dh) / 2, dw, dh);
            return c.toDataURL('image/jpeg', 0.7);
        } catch (e) { return ''; }
    }

    function saveHistory(data) {
        var entry = {
            label: data.label,
            confidence: data.confidence,
            outOfScope: data.in_scope === false,
            name: data.filename || (currentFile && currentFile.name) || 'photo',
            thumb: makeThumb(),
            time: Date.now()
        };
        var list = readHistory();
        list.unshift(entry);
        list = list.slice(0, HISTORY_MAX);
        if (!writeHistory(list)) {
            // Storage full: keep the newest few without pictures.
            list = list.slice(0, 4).map(function (x) { return { label: x.label, confidence: x.confidence, outOfScope: x.outOfScope, name: x.name, thumb: '', time: x.time }; });
            writeHistory(list);
        }
        renderHistory();
    }

    function renderHistory() {
        var list = readHistory();
        historyList.innerHTML = '';
        historySection.hidden = list.length === 0;
        list.forEach(function (item) {
            var li = document.createElement('li');
            li.className = 'history-item';
            if (item.thumb) {
                var img = document.createElement('img');
                img.src = item.thumb;
                img.alt = '';
                img.width = 64; img.height = 48;
                li.appendChild(img);
            }
            var text = document.createElement('div');
            var label = document.createElement('p');
            label.className = 'history-label';
            label.textContent = item.outOfScope ? 'Not a scene photo' : capitalize(item.label) + ' · ' + Math.round(item.confidence * 100) + '%';
            var sub = document.createElement('p');
            sub.className = 'history-sub';
            var when = new Date(item.time);
            sub.textContent = item.name + ' · ' + when.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            text.appendChild(label);
            text.appendChild(sub);
            li.appendChild(text);
            historyList.appendChild(li);
        });
    }

    historyClear.addEventListener('click', function () {
        try { localStorage.removeItem(HISTORY_KEY); } catch (e) { /* ignore */ }
        renderHistory();
    });

    // A result rendered by the server (no-JS fallback) has no preview; start clean either way.
    renderHistory();

    // Demo links: /?sample=test_image_3.jpg classifies that bundled sample straight away.
    try {
        var wanted = new URLSearchParams(window.location.search).get('sample');
        if (wanted) {
            var match = Array.prototype.find.call(document.querySelectorAll('.sample'), function (b) { return b.dataset.name === wanted; });
            if (match && !match.disabled) { match.dataset.auto = '1'; match.click(); }
        }
    } catch (e) { /* old browser: ignore */ }
})();
