import mysql.connector
from tkinter import *
import os

# Function to connect to the MySQL database
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Mysql#rjd22004",  # Your MySQL password
        database="criminal"      # Your database name
    )

# Designing window for registration
def register():
    global register_screen
    register_screen = Toplevel(main_screen)
    register_screen.title("Register")
    register_screen.geometry("300x250")

    global username
    global password
    global username_entry
    global password_entry
    username = StringVar()
    password = StringVar()

    Label(register_screen, text="Please enter details below", bg="blue").pack()
    Label(register_screen, text="").pack()
    username_label = Label(register_screen, text="Username * ")
    username_label.pack()
    username_entry = Entry(register_screen, textvariable=username)
    username_entry.pack()
    password_label = Label(register_screen, text="Password * ")
    password_label.pack()
    password_entry = Entry(register_screen, textvariable=password, show='*')
    password_entry.pack()
    Label(register_screen, text="").pack()
    Button(register_screen, text="Register", width=10, height=1, bg="blue", command=register_user).pack()

# Implementing event on register button
def register_user():
    username_info = username.get()
    password_info = password.get()

    # Connect to the database
    conn = connect_db()
    cursor = conn.cursor()

    # Insert the user details into the MySQL database
    query = "INSERT INTO users (username, password) VALUES (%s, %s)"
    values = (username_info, password_info)

    try:
        cursor.execute(query, values)
        conn.commit()
        Label(register_screen, text="Registration Success", fg="green", font=("calibri", 11)).pack()
    except Exception as e:
        conn.rollback()
        Label(register_screen, text=f"Registration Failed: {e}", fg="red", font=("calibri", 11)).pack()
    finally:
        conn.close()

    # Clear the input fields
    username_entry.delete(0, END)
    password_entry.delete(0, END)

# Designing window for login
def login():
    global login_screen
    login_screen = Toplevel(main_screen)
    login_screen.title("Login")
    login_screen.geometry("300x250")

    global username_verify
    global password_verify
    global username_login_entry
    global password_login_entry

    username_verify = StringVar()
    password_verify = StringVar()

    Label(login_screen, text="Please enter details below to login").pack()
    Label(login_screen, text="").pack()
    Label(login_screen, text="Username * ").pack()
    username_login_entry = Entry(login_screen, textvariable=username_verify)
    username_login_entry.pack()
    Label(login_screen, text="").pack()
    Label(login_screen, text="Password * ").pack()
    password_login_entry = Entry(login_screen, textvariable=password_verify, show='*')
    password_login_entry.pack()
    Label(login_screen, text="").pack()
    Button(login_screen, text="Login", width=10, height=1, command=login_verify).pack()

# Implementing event on login button
def login_verify():
    username1 = username_verify.get()
    password1 = password_verify.get()

    # Connect to the database
    conn = connect_db()
    cursor = conn.cursor()

    # SQL query to check if the user exists with the provided credentials
    query = "SELECT * FROM users WHERE username=%s AND password=%s"
    values = (username1, password1)

    cursor.execute(query, values)
    result = cursor.fetchone()

    if result:
        login_success()
    else:
        user_not_found()

    # Clear the input fields
    username_login_entry.delete(0, END)
    password_login_entry.delete(0, END)

    conn.close()

# Designing popup for login success and running home.py
def login_success():
    global login_success_screen
    login_success_screen = Toplevel(login_screen)
    login_success_screen.title("Success")
    login_success_screen.geometry("150x100")
    Label(login_success_screen, text="Login Success").pack()
    Button(login_success_screen, text="OK", command=run_home).pack()

# Function to run home.py after successful login
def run_home():
    login_success_screen.destroy()  # Close the login success popup
    main_screen.destroy()           # Close the main login window
    os.system('python "D:\MACHINE LEARNING\Facial-Recognition-for-Crime-Detection-master\Facial-Recognition-for-Crime-Detection-master\home.py"')     # Run the home.py file

# Designing popup for user not found
def user_not_found():
    global user_not_found_screen
    user_not_found_screen = Toplevel(login_screen)
    user_not_found_screen.title("Failure")
    user_not_found_screen.geometry("150x100")
    Label(user_not_found_screen, text="Invalid credentials").pack()
    Button(user_not_found_screen, text="OK", command=delete_user_not_found_screen).pack()

# Deleting popups
def delete_login_success():
    login_success_screen.destroy()

def delete_user_not_found_screen():
    user_not_found_screen.destroy()

# Designing Main (first) window
def main_account_screen():
    global main_screen
    main_screen = Tk()
    main_screen.geometry("300x250")
    main_screen.title("Account Login")
    Label(text="Select Your Choice", bg="blue", width="300", height="2", font=("Calibri", 13)).pack()
    Label(text="").pack()
    Button(text="Login", height="2", width="30", command=login).pack()
    Label(text="").pack()
    Button(text="Register", height="2", width="30", command=register).pack()

    main_screen.mainloop()

main_account_screen()
