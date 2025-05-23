from telethon import TelegramClient
from config import TELEGRAM_API_ID, TELEGRAM_API_HASH

# Используйте то же имя сессии, что и в telegram_bot.py
session_name = 'rian_listener_session'

async def main():
    print("Попытка авторизации клиента Telethon...")
    # Передаем phone=lambda: input(...) сразу в конструктор или в sign_in,
    # send_code_request не принимает phone напрямую в новых версиях, если он не был задан для клиента
    client = TelegramClient(session_name, TELEGRAM_API_ID, TELEGRAM_API_HASH)
    
    await client.connect()

    if not await client.is_user_authorized():
        print("Клиент не авторизован. Пожалуйста, введите данные для входа.")
        
        # Запрашиваем номер телефона у пользователя
        phone_number = input("Введите ваш номер телефона (в международном формате, например +7234567890): ")
        
        try:
            # Отправляем код на этот номер
            await client.send_code_request(phone_number)
            code = input("Введите код, полученный в Telegram: ")
            await client.sign_in(phone=phone_number, code=code)
        except Exception as e: # Ловим общую ошибку
            print(f"Ошибка во время ввода кода: {e}")
            # Если ошибка связана с паролем 2FA, Telethon обычно вызывает PasswordHashInvalidError или подобное,
            # но более общий Exception поймает и другие случаи.
            # Telethon сам запросит пароль, если он нужен после ввода кода.
            # Однако, если sign_in упал до запроса пароля (например, неверный код), нужно дать возможность ввести пароль.
            if "password" in str(e).lower() or "2fa" in str(e).lower() or "two-factor" in str(e).lower(): # Простая проверка
                 try:
                    password = input("Похоже, у вас включена двухфакторная аутентификация. Введите ваш пароль: ")
                    await client.sign_in(password=password)
                 except Exception as e_pw:
                    print(f"Ошибка при вводе пароля 2FA: {e_pw}")
                    return
            else:
                # Если ошибка не связана с паролем, возможно, пользователь хочет попробовать ввести пароль вручную
                try:
                    password = input("Если у вас включена двухфакторная аутентификация, введите ваш пароль (или нажмите Enter, если нет): ")
                    if password: # Если пользователь что-то ввел
                        await client.sign_in(password=password)
                except Exception as e_pw_manual:
                    print(f"Ошибка при вводе пароля 2FA (попытка вручную): {e_pw_manual}")
                    return


    if await client.is_user_authorized():
        print("Авторизация прошла успешно! Сессия сохранена.")
    else:
        print("Не удалось авторизоваться. Проверьте введенные данные и попробуйте снова.")
    
    await client.disconnect()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())