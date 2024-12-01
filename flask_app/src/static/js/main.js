
function navigate() {
    const selectElement = document.getElementById('page-selector');
    const selectedValue = selectElement.value;
    const isLogin = document.getElementById('loginCheck');
    // 페이지 이동
    if (selectedValue) {
        // 로그인 했으면 NotNull 이므로 바로 이동
        if (isLogin) {
            window.location.href = selectedValue;
        } else {
          // 로그인 하지 않은 상태이므로 null : 로그인 안내
            Swal.fire({
                title: "로그인",
                text: `로그인 후 사용해주세요`,
                icon: "warning",
                confirmButtonText : "확인"
            }).then( () => {
                window.location.href = selectedValue;
              });    
        }
        
    }
}



function sendLogoDescription() {
    const logoDescription = document.getElementById("logoDescription").value;

    // 프롬프트 입력 필수 안내
    if (logoDescription.length == 0) {
        Swal.fire({
            title: "잠깐!!",
            text: '프롬프트를 입력해주세요!',
            icon: "warning",
            confirmButtonText : "확인"
        });    
    } else {
        fetch("/logoGenerate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ description: logoDescription })
        })
        .then(response => {
    
            if (!response.ok) throw new Error("Network response was not ok");
            return response.json();
        })
        .catch(error => {
            Swal.fire({
                title: "Error",
                text: '죄송합니다. 현재 서버에서 문제가 발생했습니다.',
                icon: "error",
                confirmButtonText : "확인"
            });  
        });
    }

}



