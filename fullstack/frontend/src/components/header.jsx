import { useNavigate } from 'react-router-dom';

const Home = () => {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem('isLoggedIn');  // or localStorage.setItem('isLoggedIn', 'false');
    localStorage.removeItem('role');
    localStorage.removeItem('id');
    navigate('/login');
  };

  return (
    <div>
      <h1>Welcome Home!</h1>
      <button onClick={handleLogout} className="logout">Logout</button>
    </div>
  );
};

export default Home;
