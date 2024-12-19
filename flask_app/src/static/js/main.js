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
    const loadingSpinner = document.getElementById("loading");

    loadingSpinner.style.display = "inline-block"; // 로딩 스피너 표시 

    
    // 프롬프트 입력 필수 안내
    if (logoDescription.length == 0) {
        Swal.fire({
            title: "잠깐!!",
            text: '프롬프트를 입력해주세요!',
            icon: "warning",
            confirmButtonText : "확인"
        });    

        // 로딩 스피너 숨기기
        loadingSpinner.style.display = "none"; 

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
            window.location.href = "/myPage";
        })
        .then(data => {
            // 로딩 스피너 숨기기
            loadingSpinner.style.display = "none"; 
        })
        .catch(error => {
            Swal.fire({
                title: "Error",
                html: '죄송합니다. <br>현재 서버에서 문제가 발생했습니다.<br>계속 될 시 관리자에게 문의해주세요.',
                icon: "error",
                confirmButtonText : "확인"
            });  
        });
    }

}



const swiper = new Swiper('.swiper', {
    // Optional parameters
    autoplay: {
      delay: 5000, // 자동 재생 시간 (1초)
    },
    loop: true, // 슬라이드 무한 반복
    freeMode: true, // freemode 활성화
    pagination: {
      el: '.swiper-pagination', // pagination 요소
      clickable: true, // pagination 클릭 가능
    },
    scrollbar: {
      el: '.swiper-scrollbar', // 스크롤바 요소
      draggable: true, // 스크롤바 드래그 가능
    },
  });
  