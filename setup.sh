mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"21f3002696@student.onlinedegree.iitm.ac.in\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
