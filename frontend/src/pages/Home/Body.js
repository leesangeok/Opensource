import React from 'react';
import { useNavigate  } from 'react-router-dom'; // React Router의 useHistory 훅 import
import './body.css'; // CSS 파일을 import

const Body = () => {
    const navigate = useNavigate(); // useNavigate 훅 사용

    const handleButtonClick = () => {
      navigate('/generation'); // 버튼 클릭 시 /generation 페이지로 이동
    };
  
    return (
      <main>
        <section className="main-section">
          <h2>Welcome to the Symbol Logo Generator</h2>
          <p>Generate your custom symbol logos with our AI-powered tool.</p>
          <div className="button-container">
            <button className="btn btn-primary btn-lg" onClick={handleButtonClick}>
              이미지 생성하러 가기
            </button>
          </div>
        </section>
      </main>
    );
};

export default Body;
