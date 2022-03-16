#!/usr/local/bin/zsh


WATCHED_PID=$({ python simple_gym_jbw_agent.py $1 >log.stdout 2>log.stderr & } && echo $!);
echo "Pid:"
echo $WATCHED_PID
while ps -p $WATCHED_PID -o "etime= pid= %cpu= %mem= rss="; do 
   sleep 1 
done
