function navigate() {
    const selectElement = document.getElementById('page-selector');
    const selectedValue = selectElement.value;

    // 페이지 이동
    if (selectedValue) {
        // 페이지 이동
        window.location.href = selectedValue;
    }
}

$(document).ready(function(){
    $('[data-toggle="tooltip"]').tooltip();
});