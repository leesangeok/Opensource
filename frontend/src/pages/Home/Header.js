import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './header.css';

const Header = () => {
  const navigate = useNavigate(); // 페이지 이동을 위한 hook
  const location = useLocation(); // 현재 경로 확인을 위한 hook
  const [selectedPage, setSelectedPage] = useState('/'); // 선택된 페이지 상태

  // 경로가 변경될 때 콤보박스 값을 자동으로 변경
  useEffect(() => {
    setSelectedPage(location.pathname); // 현재 경로로 콤보박스 상태 업데이트
  }, [location.pathname]);

  const handleLogoClick = () => {
    navigate('/'); // 메인 페이지로 이동
  };

  const handlePageChange = (event) => {
    navigate(event.target.value); // 선택한 페이지로 이동
  };

  return (
    <header>
      <div className="header-container">
        {/* 로고 클릭 시 메인 페이지로 이동 */}
        <h1 onClick={handleLogoClick} className="clickable-logo">LogoGen</h1>
        
        {/* 콤보박스 추가, 현재 경로에 따라 선택된 값이 변경되도록 함 */}
        <select value={selectedPage} onChange={handlePageChange} className="page-selector">
          <option value="/">Home</option>
          <option value="/generation">Image Generation</option>
          <option value="/mypage">MyPage</option>
        </select>
      </div>
    </header>
  );
};

export default Header;
