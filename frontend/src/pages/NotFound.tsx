import { Link } from "react-router-dom";

export default function NotFound() {
    return (
        <div className="page-container not-found-page">
            <div className="not-found-content">
                <h1 className="not-found-code">404</h1>
                <h2 className="not-found-title">Page Not Found</h2>
                <p className="not-found-message">
                    Looks like this pitch went wide. The page you're looking for doesn't exist.
                </p>
                <Link to="/" className="btn-cta">
                    Back to Home
                </Link>
            </div>
        </div>
    );
}
