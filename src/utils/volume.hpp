#pragma once

#include <fstream>
#include <string>
#include <memory>

#include <utils/attribute.hpp>

namespace vol
{
template <typename Voxel>
struct Volume;

struct VolumeBlockDim
{
	VOL_DEFINE_ATTRIBUTE( unsigned, width );
	VOL_DEFINE_ATTRIBUTE( unsigned, height );
	VOL_DEFINE_ATTRIBUTE( unsigned, depth );

public:
	std::size_t size() const { return std::size_t( width ) * height * depth; }
};

template <typename Voxel>
struct VolumeBlock
{
private:
	struct Inner
	{
		std::string _;
	};

public:
	const Voxel *data() const
	{
		return reinterpret_cast<const Voxel *>( _->_.c_str() );
	}
	Voxel *data()
	{
		return reinterpret_cast<Voxel *>( const_cast<char *>( _->_.c_str() ) );
	}

private:
	VolumeBlock( std::string &&_ ) { this->_->_ = std::move( _ ); }

	std::shared_ptr<Inner> _ = std::make_shared<Inner>();
	friend struct Volume<Voxel>;
};

template <typename Voxel>
struct Volume
{
	static Volume from_raw( const std::string &file_name,
							VolumeBlockDim const &dim )
	{
		Volume vol( std::ifstream( file_name, std::ios::in | std::ios::binary ), dim );
		vol.cnt = 1;
		return std::move( vol );
	}

	VolumeBlock<Voxel> get_block( std::size_t idx )
	{
		std::string buffer;
		auto block_size = sizeof( Voxel ) * dim.size();
		buffer.resize( block_size );
		auto buffer_ptr = const_cast<char *>( buffer.c_str() );
		auto nread = _.seekg( offset + idx * block_size )
					   .read( buffer_ptr, block_size )
					   .gcount();
		if ( nread != block_size ) {
			throw std::runtime_error( "failed to read block" );
		}
		return VolumeBlock<Voxel>( std::move( buffer ) );
	}

	std::size_t block_count() const { return cnt; }

private:
	Volume( std::ifstream &&_, VolumeBlockDim const &dim ) :
	  _( std::move( _ ) ),
	  dim( dim ) {}

private:
	std::ifstream _;
	std::size_t cnt;
	std::size_t offset = 0;
	VolumeBlockDim dim;
};
}  // namespace vol
