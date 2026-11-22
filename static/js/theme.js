// Light / dark theme toggle. Follows the system setting until the user picks one.
(function () {
    var root = document.documentElement;
    var btn = document.getElementById('theme-toggle');
    if (!btn) return;
    var media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

    function current() {
        var t = root.getAttribute('data-theme');
        if (t === 'light' || t === 'dark') return t;
        return media && media.matches ? 'dark' : 'light';
    }
    function label() {
        var next = current() === 'dark' ? 'light' : 'dark';
        btn.setAttribute('aria-label', 'Switch to ' + next + ' theme');
        btn.title = 'Switch to ' + next + ' theme';
    }
    btn.addEventListener('click', function () {
        var next = current() === 'dark' ? 'light' : 'dark';
        root.setAttribute('data-theme', next);
        try { localStorage.setItem('scene-theme', next); } catch (e) { /* storage blocked */ }
        label();
    });
    if (media && media.addEventListener) media.addEventListener('change', label);
    label();
})();
