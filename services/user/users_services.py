from models.users import Users as users_models
from schemas import users_schemas as users_schemas
from sqlalchemy.ext.asyncio import AsyncSession
import datetime
from fastapi import HTTPException,status
from sqlalchemy.future import select
from typing import List
from core.security import get_password_hash
from core.auth import authenticate_user
import secrets
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sqlalchemy import func
from models.roles import Roles as roles_models


async def login_user(cpf:str,password:str,db:AsyncSession):
    user = await authenticate_user(cpf,password,db)
    return user
    
async def register_user(user:users_schemas.users_create,db:AsyncSession) -> users_models: 
    new_user:users_models=users_models(name =user.name,email=user.email,
                                       cpf=user.cpf,phone=user.phone,password=get_password_hash(user.password))
    async with db as session:
            session.add(new_user)
            await session.commit()
            await session.refresh(new_user)
            return new_user

async def select_all_users(db:AsyncSession) -> List[users_schemas.usersGetData]:
    async with db as session:
        querie = select(users_models).order_by(users_models.id.asc()).filter(users_models.active==True)
        resultset = await session.execute(querie)
        users:List[users_schemas.usersGetData] = resultset.scalars().unique().all()
        
        users_list = []
        for user in users:
            querie = select(func.count(routines_models.id)).filter(routines_models.users_id == user.id,routines_models.active == True)
            resultset = await session.execute(querie)
            count = resultset.scalar()
            
            #role
            role = select(roles_models).filter(roles_models.id == user.role_id,roles_models.active==True)
            role = await session.execute(role)
            role = role.scalars().unique().one_or_none()
            
            users_list.append(
                {
                    "id":user.id,
                    "name":user.name,
                    "email":user.email,
                    "cpf":user.cpf,
                    "phone":user.phone,
                    "active":user.active,
                    "role":role.get_role_display(),
                    "tarefas":count
                }
            )
        
        
        return users_list
    
async def select_user(id_user:int,db:AsyncSession) -> users_schemas.users:
    async with db as session:
        querie = select(users_models).filter(users_models.id==id_user,users_models.active==True)
        resultset = await session.execute(querie)
        user = resultset.scalars().unique().one_or_none()
        
        return user

async def update_user(id_user:int,user:users_schemas.users_update,db:AsyncSession) -> bool:
    async with db as session:
        querie = select(users_models).filter(users_models.id == id_user,users_models.active==True)
        resultset = await session.execute(querie)
        user_up:users_schemas.users = resultset.scalars().unique().one_or_none()
        
        if user_up:
            if user["name"]:
                user_up.name = user['name']
            if user["email"]:
                user_up.email = user['email']
            if user['active'] is not None:
                user_up.active = user['active']
            if user['phone']:
                user_up.phone = user['phone']
            if user['cpf']: 
                user_up.cpf = user['cpf'] 
            await session.commit() 
            return True
        return False
            
    
async def drop_user(id_user:int, db:AsyncSession):
    async with db as session:
        querie = select(users_models).filter(users_models.id==int(id_user),users_models.active==True)
        result_set = await session.execute(querie)
        user_delete:users_schemas.users = result_set.scalars().unique().one_or_none()
        if user_delete:
            user_delete.active = False
            await session.commit()
            await session.refresh(user_delete)
            return user_delete
            
async def get_user_by_email(email:str,db:AsyncSession):
    async with db as session:
        querie = select(users_models).filter(users_models.email==email,users_models.active==True)
        resultset = await session.execute(querie)
        user_up:users_schemas.users = resultset.scalars().unique().one_or_none()
        if user_up:
            return user_up
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    
async def generate_reset_token(email:str,db:AsyncSession):
    async with db as session:
        user:users_schemas.users = await get_user_by_email(email,db)
        token = secrets.token_urlsafe(16)
        user.reset_password_token = token
        user.reset_password_expires = datetime.datetime.now() + datetime.timedelta(hours=1)
        session.add(user)
        await session.commit()
        return token

async def send_email(email: str,token:str):
    # configuração do servidor de email
    sender_email = "gabrieldrumond211@gmail.com"
    receiver_email = email
    password = "jggp jojt pcop pjub"
    
    message = MIMEMultipart("alternative")
    message["Subject"] = "Recuperação de senha"
    message["From"] = f"Cod3Bit Dev Team <{sender_email}>"
    message["To"] = receiver_email
    
    reset_link = f"http://127.0.0.1:3000/resetpassword?email={email}&token={token}"
    text = f"""\
    Olá,
    Recebemos uma solicitação para redefinir sua senha. Clique no link abaixo para redefinir sua senha:
    {reset_link}
    Se você não solicitou a redefinição de senha, ignore este e-mail.
    Obrigado,
    Cod3Bit - Development Team
    """
    part = MIMEText(text, "plain")
    message.attach(part)
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:  # Inicia a conexão TLS
            server.login(user=sender_email,password=password)
            server.sendmail(sender_email, receiver_email, message.as_string())
    except smtplib.SMTPException as e:
        print(f"Erro SMTP: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Erro ao enviar e-mail.")
    except Exception as e:
        print(f"Erro geral: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Erro ao enviar e-mail.")
    
    return True
    
async def get_user_by_reset_token(token: str, db: AsyncSession):
    async with db as session:
        query = select(users_models).filter(users_models.reset_password_token == token,users_models.active==True)
        resultset = await session.execute(query)
        user_up: users_schemas.users_update = resultset.scalars().unique().one_or_none()
        if user_up:
            return user_up
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Token inválido ou expirado")
            
        
        
        


