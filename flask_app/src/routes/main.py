import os
import requests
from routes.kakao import kakaoLogout
from decorators import login_required
from flask import Flask, Blueprint, request, redirect, jsonify,render_template, session

main = Blueprint('main', __name__)

@main.route('/')
def hello():

    return render_template('main.html')

@main.route('/login')
def login():
    return render_template('login.html')

@main.route('/logout')
def logout():

    if session['user_id'] :
        kakaoLogout(session.pop('access_token'),
                    session.pop('user_id'))
    
        return redirect('/')

@main.route('/generate')
@login_required
def generateLogoImage():
    return render_template('generation_page.html')
    
@main.route('/myPage')
@login_required
def myPage():
    return render_template('myPage.html')
