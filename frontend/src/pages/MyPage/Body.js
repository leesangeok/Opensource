import React from 'react';
import './mypage_body.css'; // CSS 파일을 import
import { Box, Typography, Paper, Button } from '@mui/material'; // MUI에서 컴포넌트 임포트
import hknuImage from '../../hknu.jpg'; // 이미지 경로 임포트

const Body = () => {
  // 여러 장의 이미지를 배열로 설정 (같은 이미지를 반복해서 사용)
  const images = [
    hknuImage,
    hknuImage,
    hknuImage,
    hknuImage,
    hknuImage
  ];

  return (
    <main className="mypage-main-section">
      <Paper className="content-box" elevation={3}>
        <Typography variant="body1" className="page-description"sx={{ fontSize: '1.5rem', fontFamily: 'Roboto, sans-serif', fontWeight: '500', color: '#333' }}>저장된 이미지를 확인하세요.</Typography>
        {/* 이미지들을 스크롤 가능한 영역에 넣음 */}
        <Box className="image-gallery">
          {images.map((image, index) => (
            <img
              key={index}
              src={image}
              alt={`Hankuk National University ${index + 1}`}
              className="gallery-image"
            />
          ))}
        </Box>


      </Paper>
    </main>
  );
};

export default Body;
