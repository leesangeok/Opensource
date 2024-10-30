import React from 'react';
import './footer.css';

const Footer = () => {
  return (
    <footer>
      <div className="footer-container">
        <p>&copy; 2024 OpenSource Project Logo Generator</p>
        <p>조원: 김기현, 김민근, 김민수, 이상억, 허찬욱</p>
        <nav>
          <ul>
            <li>
              <a href="https://github.com/leesangeok/Opensource.git" target="_blank" rel="noopener noreferrer">
                Github: https://github.com/leesangeok/Opensource.git
              </a>
            </li>
          </ul>
        </nav>
      </div>
    </footer>
  );
};

export default Footer;
