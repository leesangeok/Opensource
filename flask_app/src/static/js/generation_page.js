document.getElementById('manual').addEventListener('click', function () {
    setTimeout(function () {
        document.getElementById('collapseExample').scrollIntoView({
            behavior: 'smooth'
        });
    }, 300);
});