import { createContext, useContext } from 'react';

// Create the context
const AuthContext = createContext();

// Create the hook
export const useAuth = () => useContext(AuthContext);

// Export the context for use in the provider
export { AuthContext };

// Export a default object to allow default imports if needed
export default { AuthContext, useAuth };
