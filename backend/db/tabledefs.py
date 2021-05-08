from sqlalchemy import Column, Integer, DECIMAL, String, DateTime
from sqlalchemy_utils import IPAddressType
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()


class Relation(Base):
    __tablename__ = 'relation'
    relationid = Column('relationid', Integer, primary_key=True)
    relationgroupid = Column('relationgroupid', Integer)
    relation = Column('relation', String)
    score = Column('score', DECIMAL)


class RelationGroup(Base):
    __tablename__ = 'relationgroup'
    relationgroupid = Column('relationgroupid', Integer, primary_key=True)
    cue = Column('cue', String)
    modelname = Column('modelname', String)
    created = Column('created', DateTime, default=datetime.datetime.utcnow)


class Response(Base):
    __tablename__ = 'response'
    responseid = Column('responseid', Integer, primary_key=True)
    relationid = Column('relationid', Integer)
    timeduration = Column('timeduration', Integer)  # TODO is the time duration for each word individually?
    responsegroupid = Column('responsegroupid', Integer)


class ResponseGroup(Base):
    __tablename__ = 'responsegroup'
    responsegroupid = Column('responsegroupid', Integer, primary_key=True)
    userid = Column('userid', Integer)
    relationtype = Column('relationtype', String)
    relationgroupid = Column('relationgroupid', Integer)
    created = Column('created', DateTime)
    ipaddress = Column('ipaddress', IPAddressType)
    useragentlocaleid = Column('useragentlocaleid', Integer)


class Users(Base):
    __tablename__ = 'users'
    userid = Column('userid', Integer, primary_key=True)
    external_user_id = Column('external_user_id', String)
    age = Column('age', Integer)
    nativelanguage = Column('nativelanguage', Integer)
    gender = Column('gender', String)
    city = Column('city', String)
    country = Column('country', String)
    source = Column('source', String)


class UserAgentLocale(Base):
    __tablename__ = 'useragentlocale'
    useragentlocaleid = Column('useragentlocaleid', Integer, primary_key=True)
    useragent = Column('useragent', String)
    locale = Column('locale', String)
