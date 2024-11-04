
// 페이지가 로드될 때 선택된 값을 설정
window.onload = function() {
    const savedValue = localStorage.getItem('selectedPage');
    if (savedValue) {
        document.getElementById('page-selector').value = savedValue;
    }
};

function navigate() {
    const selectElement = document.getElementById('page-selector');
    const selectedValue = selectElement.value;

    // 선택된 값을 localStorage에 저장
    localStorage.setItem('selectedPage', selectedValue);

    // 페이지 이동
    window.location.href = selectedValue;
}
