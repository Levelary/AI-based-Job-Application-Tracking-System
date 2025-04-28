import { useNavigate } from 'react-router-dom';

const Home = () => {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.clear();
    // window.location.href = '/login';
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
