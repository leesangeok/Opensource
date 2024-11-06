function navigate() {
    const selectElement = document.getElementById('page-selector');
    const selectedValue = selectElement.value;

    // 페이지 이동
    if (selectedValue) {
        // 페이지 이동
        window.location.href = selectedValue;
    }
}



function sendLogoDescription() {
    const logoDescription = document.getElementById("logoDescription").value;
    
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
        console.error("Error:", error);
    });
}