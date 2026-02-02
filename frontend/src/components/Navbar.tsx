import { NavLink } from 'react-router-dom';
import '../styles/Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-container">
        <NavLink to="/" className="navbar-logo">

          <span className="logo-text">Dugout<span className="logo-accent">Data</span></span>
        </NavLink>
        <ul className="nav-menu">
          <li className="nav-item">
            <NavLink to="/" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"} end>
              Home
            </NavLink>
          </li>
          <li className="nav-item">
            <NavLink to="/predictions" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
              Predictions
            </NavLink>
          </li>
          <li className="nav-item">
            <NavLink to="/search" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
              Player Search
            </NavLink>
          </li>
          {/* <li className="nav-item">
            <NavLink to="/next-season" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
              Next Season
            </NavLink>
          </li> */}
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;