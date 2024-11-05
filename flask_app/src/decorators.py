# decorators.py
from functools import wraps
from flask import session, redirect

# 로그인 체크 
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or 'access_token' not in session:
            return redirect('/')  
        return f(*args, **kwargs)
    return decorated_function
