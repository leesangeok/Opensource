document.getElementById('manual').addEventListener('click', function () {
    setTimeout(function () { // Collapse가 열릴 시간을 기다림
        document.getElementById('collapseExample').scrollIntoView({
            behavior: 'smooth'
        });
    }, 300); // Collapse 애니메이션이 끝날 때까지 약간의 딜레이 (기본값 300ms)
});

